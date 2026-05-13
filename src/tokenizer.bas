' *******************************************************
' * Tokenizer for GPT-2 BASIC                          *
' *******************************************************
' * This module implements a simplified tokenizer for   *
' * the GPT-2 model, adapted for 486-era hardware       *
' * constraints.                                        *
' *                                                     *
' * It provides simplified BPE (Byte-Pair Encoding)     *
' * functionality and vocabulary management, optimized  *
' * for lower memory usage.                             *
' *******************************************************

#INCLUDE "src/data_structures.bas"

' *******************************************************
' * Constants and Data Structures                       *
' *******************************************************

' Maximum vocabulary size supported
CONST MAX_VOCAB_SIZE = 16384 ' Reduced from 50257 in original GPT-2

' Maximum merge operations for BPE
CONST MAX_MERGES = 16384 ' Reduced from ~40000 in original GPT-2

' Maximum token length in bytes
CONST MAX_TOKEN_LENGTH = 16

' Built-in byte vocabulary size: EOT, UNK, and 256 byte tokens
CONST BYTE_VOCAB_SIZE = 258

' Hash table capacity for token lookup. External vocabularies larger than this
' need a wider table before they can be loaded safely.
CONST TOKEN_HASH_SIZE = 4096

' Optional VOCAB.BIN extension marker for per-token output sampling masks.
CONST OUTPUT_MASK_MAGIC = &H4B53414D
' Optional VOCAB.BIN extension marker for tokenizer mode.
CONST TOKENIZER_MODE_MAGIC = &H45444F4D
CONST TOKENIZER_MODE_BYTE = 0
CONST TOKENIZER_MODE_BPE = 1
CONST TOKENIZER_MODE_LEXICON = 2

' End of text token
CONST EOT_TOKEN = 0
' Unknown token
CONST UNK_TOKEN = 1

' Vocabulary entry structure
TYPE VocabEntry
    token AS STRING * MAX_TOKEN_LENGTH
    token_len AS INTEGER ' Actual length of the token string
    token_id AS INTEGER
    count AS LONG ' For frequency-based operations
END TYPE

' Merge operation for BPE
TYPE MergeOp
    first AS INTEGER ' Index of first token in merge
    second AS INTEGER ' Index of second token in merge
    result AS INTEGER ' Index of resulting merged token
    priority AS INTEGER ' Priority/rank of this merge
END TYPE

' Tokenizer structure
TYPE Tokenizer
    vocab(0 TO MAX_VOCAB_SIZE - 1) AS VocabEntry
    vocab_size AS INTEGER
    merges(0 TO MAX_MERGES - 1) AS MergeOp
    merge_count AS INTEGER
    output_allowed(0 TO MAX_VOCAB_SIZE - 1) AS BYTE
    vocab_hash(0 TO TOKEN_HASH_SIZE - 1) AS INTEGER ' Simple hash table for token lookup
    lexicon_bucket_start(0 TO 255) AS INTEGER ' First-byte buckets for fast longest-match lexicon scans
    lexicon_bucket_count(0 TO 255) AS INTEGER
    lexicon_length_present(0 TO 255, 1 TO MAX_TOKEN_LENGTH) AS BYTE
    lexicon_order(0 TO MAX_VOCAB_SIZE - 1) AS INTEGER
    tokenizer_mode AS INTEGER ' byte, BPE, or longest-match lexicon
    use_bpe AS INTEGER ' Whether to use BPE or just byte-level tokenization
    max_token_length AS INTEGER
    cache_enabled AS INTEGER
END TYPE

' Global tokenizer instance
DIM SHARED g_tokenizer AS Tokenizer

DECLARE FUNCTION AddToken(BYREF tokenizer AS Tokenizer, token AS STRING, token_id AS INTEGER) AS INTEGER
DECLARE FUNCTION HashToken(token AS STRING) AS INTEGER
DECLARE FUNCTION CleanTokenizerText(input_text AS STRING) AS STRING
DECLARE FUNCTION TokenizerOutputAllowed(tokenizer AS Tokenizer, token_id AS INTEGER) AS INTEGER
DECLARE FUNCTION TokenizerLexiconWordByte(byte_value AS INTEGER) AS INTEGER
DECLARE FUNCTION TokenizerLexiconBoundaryNextOK(bytes() AS BYTE, next_idx AS INTEGER, byte_count AS INTEGER) AS INTEGER
DECLARE FUNCTION TokenPieceMatchesBytes(tokenizer AS Tokenizer, vocab_idx AS INTEGER, bytes() AS BYTE, start_idx AS INTEGER, byte_count AS INTEGER) AS INTEGER
DECLARE FUNCTION TokenPieceBoundaryMatches(tokenizer AS Tokenizer, vocab_idx AS INTEGER, bytes() AS BYTE, start_idx AS INTEGER, byte_count AS INTEGER) AS INTEGER
DECLARE SUB TokenizerBuildLexiconBuckets(BYREF tokenizer AS Tokenizer)
DECLARE SUB BPETokenize(tokenizer AS Tokenizer, bytes() AS BYTE, byte_count AS INTEGER, tokens() AS INTEGER, BYREF token_count AS INTEGER)
DECLARE SUB LexiconTokenize(tokenizer AS Tokenizer, bytes() AS BYTE, byte_count AS INTEGER, tokens() AS INTEGER, BYREF token_count AS INTEGER)

' *******************************************************
' * Initialization and Setup                            *
' *******************************************************

' Initialize a tokenizer
SUB InitTokenizer(BYREF tokenizer AS Tokenizer)
    DIM i AS INTEGER
    DIM added_index AS INTEGER

    ' Clear tokenizer data
    tokenizer.vocab_size = 0
    tokenizer.merge_count = 0
    tokenizer.tokenizer_mode = TOKENIZER_MODE_BYTE
    tokenizer.use_bpe = 0 ' Byte-level default; external vocab enables BPE when merges exist.
    tokenizer.max_token_length = MAX_TOKEN_LENGTH
    tokenizer.cache_enabled = 1 ' Enable cache by default

    ' Initialize hash table to -1 (empty)
    FOR i = 0 TO TOKEN_HASH_SIZE - 1
        tokenizer.vocab_hash(i) = -1
    NEXT i
    FOR i = 0 TO 255
        tokenizer.lexicon_bucket_start(i) = 0
        tokenizer.lexicon_bucket_count(i) = 0
        FOR added_index = 1 TO MAX_TOKEN_LENGTH
            tokenizer.lexicon_length_present(i, added_index) = 0
        NEXT added_index
    NEXT i
    FOR i = 0 TO MAX_VOCAB_SIZE - 1
        tokenizer.output_allowed(i) = 0
        tokenizer.lexicon_order(i) = -1
    NEXT i

    ' Initialize with basic tokens (special tokens and ASCII chars)
    ' Add special tokens
    added_index = AddToken(tokenizer, "<|endoftext|>", EOT_TOKEN)
    added_index = AddToken(tokenizer, "<|unk|>", UNK_TOKEN)

    ' Add basic ASCII characters (0-255)
    FOR i = 0 TO 255
        DIM char_token AS STRING * 2
        char_token = CHR$(i)
        added_index = AddToken(tokenizer, char_token, i + 2) ' +2 because 0,1 are special
    NEXT i
    tokenizer.output_allowed(EOT_TOKEN) = 1
    tokenizer.output_allowed(UNK_TOKEN) = 0
    FOR i = 32 TO 126
        tokenizer.output_allowed(i + 2) = 1
    NEXT i

    PRINT "Initialized tokenizer with "; tokenizer.vocab_size; " tokens"
END SUB

' Add a token to the vocabulary
FUNCTION AddToken(BYREF tokenizer AS Tokenizer, token AS STRING, token_id AS INTEGER) AS INTEGER
    DIM i AS INTEGER, hash_val AS INTEGER, probe_count AS INTEGER

    ' Check if we have room
    IF tokenizer.vocab_size >= MAX_VOCAB_SIZE THEN
        PRINT "ERROR: Vocabulary is full"
        RETURN -1
    END IF
    IF tokenizer.vocab_size >= TOKEN_HASH_SIZE THEN
        PRINT "ERROR: Token hash table is full"
        RETURN -1
    END IF

    ' Check if token already exists
    hash_val = HashToken(token) MOD TOKEN_HASH_SIZE
    i = tokenizer.vocab_hash(hash_val)
    probe_count = 0

    WHILE i >= 0 AND probe_count < TOKEN_HASH_SIZE
        IF LEFT$(tokenizer.vocab(i).token, tokenizer.vocab(i).token_len) = token THEN
            ' Token already exists
            RETURN i
        END IF
        ' Linear probing for collision
        hash_val = (hash_val + 1) MOD TOKEN_HASH_SIZE
        i = tokenizer.vocab_hash(hash_val)
        probe_count = probe_count + 1
    WEND
    IF probe_count >= TOKEN_HASH_SIZE THEN
        PRINT "ERROR: Token hash table lookup failed"
        RETURN -1
    END IF

    ' Add new token
    i = tokenizer.vocab_size
    tokenizer.vocab(i).token = token
    tokenizer.vocab(i).token_len = LEN(token)
    tokenizer.vocab(i).token_id = token_id
    tokenizer.vocab(i).count = 0
    IF token_id >= 0 AND token_id < MAX_VOCAB_SIZE THEN
        tokenizer.output_allowed(token_id) = 1
        IF token_id = UNK_TOKEN THEN tokenizer.output_allowed(token_id) = 0
        IF token_id >= 2 AND token_id < BYTE_VOCAB_SIZE THEN
            IF token_id - 2 < 32 OR token_id - 2 > 126 THEN
                tokenizer.output_allowed(token_id) = 0
            END IF
        END IF
    END IF

    ' Update hash table
    hash_val = HashToken(token) MOD TOKEN_HASH_SIZE
    probe_count = 0
    WHILE tokenizer.vocab_hash(hash_val) >= 0 AND probe_count < TOKEN_HASH_SIZE
        ' Linear probing for collision
        hash_val = (hash_val + 1) MOD TOKEN_HASH_SIZE
        probe_count = probe_count + 1
    WEND
    IF probe_count >= TOKEN_HASH_SIZE THEN
        PRINT "ERROR: Token hash table insert failed"
        RETURN -1
    END IF
    tokenizer.vocab_hash(hash_val) = i

    ' Increment vocabulary size
    tokenizer.vocab_size = tokenizer.vocab_size + 1

    RETURN i
END FUNCTION

' Simple string hash function
FUNCTION HashToken(token AS STRING) AS INTEGER
    DIM hash_val AS LONG, i AS INTEGER

    hash_val = 0
    FOR i = 1 TO LEN(token)
        hash_val = (hash_val * 31 + ASC(MID$(token, i, 1))) AND &H7FFFFFFF
    NEXT i

    RETURN hash_val
END FUNCTION

' Find a token in the vocabulary
FUNCTION FindToken(tokenizer AS Tokenizer, token AS STRING) AS INTEGER
    DIM hash_val AS INTEGER, i AS INTEGER, probe_count AS INTEGER

    hash_val = HashToken(token) MOD TOKEN_HASH_SIZE
    i = tokenizer.vocab_hash(hash_val)
    probe_count = 0

    WHILE i >= 0 AND probe_count < TOKEN_HASH_SIZE
        IF LEFT$(tokenizer.vocab(i).token, tokenizer.vocab(i).token_len) = token THEN
            RETURN i
        END IF
        ' Linear probing for collision
        hash_val = (hash_val + 1) MOD TOKEN_HASH_SIZE
        i = tokenizer.vocab_hash(hash_val)
        probe_count = probe_count + 1
    WEND

    ' Token not found
    RETURN -1
END FUNCTION

' Normalize prompt/document text the same way as the host tokenizer: ASCII only,
' printable characters only, collapsed whitespace, and no leading/trailing space.
FUNCTION CleanTokenizerText(input_text AS STRING) AS STRING
    DIM result AS STRING
    DIM i AS INTEGER
    DIM char_code AS INTEGER
    DIM pending_space AS INTEGER

    result = ""
    pending_space = 0

    FOR i = 1 TO LEN(input_text)
        char_code = ASC(MID$(input_text, i, 1))
        IF char_code >= 33 AND char_code <= 126 THEN
            IF pending_space <> 0 AND LEN(result) > 0 THEN
                result = result + " "
            END IF
            result = result + CHR$(char_code)
            pending_space = 0
        ELSEIF char_code = 9 OR char_code = 10 OR char_code = 13 OR char_code = 32 THEN
            IF LEN(result) > 0 THEN
                pending_space = 1
            END IF
        END IF
    NEXT i

    RETURN result
END FUNCTION

' *******************************************************
' * Vocabulary Management                               *
' *******************************************************

' Load vocabulary from a file
SUB LoadVocabulary(BYREF tokenizer AS Tokenizer, filename AS STRING)
    DIM file AS LONG, i AS INTEGER, j AS INTEGER
    DIM token AS STRING * MAX_TOKEN_LENGTH
    DIM token_len AS INTEGER
    DIM token_id AS INTEGER
    DIM loaded_vocab_size AS INTEGER
    DIM loaded_merge_count AS INTEGER
    DIM hash_val AS INTEGER
    DIM probe_count AS INTEGER
    DIM expected_byte AS INTEGER
    DIM byte_found AS INTEGER
    DIM error_message AS STRING
    DIM extension_marker AS LONG
    DIM mask_count AS INTEGER
    DIM mask_value AS BYTE
    DIM mode_code AS INTEGER
    DIM mode_seen AS INTEGER
    DIM mask_seen AS INTEGER
    DIM token_id_seen(0 TO MAX_VOCAB_SIZE - 1) AS BYTE

    ' Open vocabulary file
    file = FREEFILE
    OPEN filename FOR BINARY AS file

    FOR i = 0 TO TOKEN_HASH_SIZE - 1
        tokenizer.vocab_hash(i) = -1
    NEXT i
    FOR i = 0 TO MAX_VOCAB_SIZE - 1
        token_id_seen(i) = 0
        tokenizer.output_allowed(i) = 0
    NEXT i
    tokenizer.tokenizer_mode = TOKENIZER_MODE_BYTE
    tokenizer.use_bpe = 0
    mode_seen = 0
    mask_seen = 0

    ' Read the vocabulary size
    GET #file, , loaded_vocab_size
    IF loaded_vocab_size < BYTE_VOCAB_SIZE THEN
        error_message = "vocab size below byte vocabulary"
        GOTO load_failed
    END IF
    IF loaded_vocab_size > MAX_VOCAB_SIZE THEN
        error_message = "vocab size exceeds runtime maximum"
        GOTO load_failed
    END IF
    IF loaded_vocab_size > TOKEN_HASH_SIZE THEN
        error_message = "vocab size exceeds token hash table capacity"
        GOTO load_failed
    END IF

    ' Read token_size tokens
    FOR i = 0 TO loaded_vocab_size - 1
        ' Read token length
        GET #file, , token_len
        IF token_len <= 0 OR token_len > MAX_TOKEN_LENGTH THEN
            error_message = "invalid token length"
            GOTO load_failed
        END IF

        ' Read token string (fixed size)
        GET #file, , token

        ' Read token ID
        GET #file, , token_id
        IF token_id < 0 OR token_id >= loaded_vocab_size THEN
            error_message = "token id outside vocabulary size"
            GOTO load_failed
        END IF
        IF token_id_seen(token_id) <> 0 THEN
            error_message = "duplicate token id"
            GOTO load_failed
        END IF
        token_id_seen(token_id) = 1

        ' Store in vocabulary
        tokenizer.vocab(i).token = token
        tokenizer.vocab(i).token_len = token_len
        tokenizer.vocab(i).token_id = token_id
        tokenizer.vocab(i).count = 0
        tokenizer.output_allowed(token_id) = 1

        ' Update hash table
        hash_val = HashToken(LEFT$(token, token_len)) MOD TOKEN_HASH_SIZE
        probe_count = 0
        WHILE tokenizer.vocab_hash(hash_val) >= 0 AND probe_count < TOKEN_HASH_SIZE
            ' Linear probing for collision
            hash_val = (hash_val + 1) MOD TOKEN_HASH_SIZE
            probe_count = probe_count + 1
        WEND
        IF probe_count >= TOKEN_HASH_SIZE THEN
            error_message = "token hash table insert failed"
            GOTO load_failed
        END IF
        tokenizer.vocab_hash(hash_val) = i
    NEXT i
    tokenizer.vocab_size = loaded_vocab_size

    FOR expected_byte = 0 TO 255
        byte_found = 0
        FOR j = 0 TO loaded_vocab_size - 1
            IF tokenizer.vocab(j).token_id = expected_byte + 2 THEN
                byte_found = 1
                IF tokenizer.vocab(j).token_len <> 1 THEN
                    error_message = "byte token length mismatch"
                    GOTO load_failed
                END IF
                IF ASC(LEFT$(tokenizer.vocab(j).token, 1)) <> expected_byte THEN
                    error_message = "byte token value mismatch"
                    GOTO load_failed
                END IF
                EXIT FOR
            END IF
        NEXT j
        IF byte_found = 0 THEN
            error_message = "missing byte token"
            GOTO load_failed
        END IF
    NEXT expected_byte

    ' Read merge operations
    GET #file, , loaded_merge_count
    IF loaded_merge_count < 0 OR loaded_merge_count > MAX_MERGES THEN
        error_message = "merge count outside runtime maximum"
        GOTO load_failed
    END IF
    IF loaded_vocab_size = BYTE_VOCAB_SIZE AND loaded_merge_count > 0 THEN
        error_message = "byte vocabulary cannot contain BPE merges"
        GOTO load_failed
    END IF

    tokenizer.merge_count = loaded_merge_count
    FOR i = 0 TO loaded_merge_count - 1
        GET #file, , tokenizer.merges(i).first
        GET #file, , tokenizer.merges(i).second
        GET #file, , tokenizer.merges(i).result
        GET #file, , tokenizer.merges(i).priority
        IF tokenizer.merges(i).first < 2 OR tokenizer.merges(i).first >= loaded_vocab_size THEN
            error_message = "merge first token outside vocabulary"
            GOTO load_failed
        END IF
        IF tokenizer.merges(i).second < 2 OR tokenizer.merges(i).second >= loaded_vocab_size THEN
            error_message = "merge second token outside vocabulary"
            GOTO load_failed
        END IF
        IF tokenizer.merges(i).result < BYTE_VOCAB_SIZE OR tokenizer.merges(i).result >= loaded_vocab_size THEN
            error_message = "merge result outside BPE vocabulary"
            GOTO load_failed
        END IF
        IF tokenizer.merges(i).priority < 0 THEN
            error_message = "negative merge priority"
            GOTO load_failed
        END IF
    NEXT i

    tokenizer.output_allowed(EOT_TOKEN) = 1
    tokenizer.output_allowed(UNK_TOKEN) = 0
    FOR i = 0 TO 255
        IF i >= 32 AND i <= 126 THEN
            tokenizer.output_allowed(i + 2) = 1
        ELSE
            tokenizer.output_allowed(i + 2) = 0
        END IF
    NEXT i

    WHILE SEEK(file) <= LOF(file)
        GET #file, , extension_marker
        IF extension_marker = OUTPUT_MASK_MAGIC THEN
            IF mask_seen <> 0 THEN
                error_message = "duplicate output mask extension"
                GOTO load_failed
            END IF
            mask_seen = 1
            GET #file, , mask_count
            IF mask_count <> loaded_vocab_size THEN
                error_message = "output mask count does not match vocabulary"
                GOTO load_failed
            END IF
            FOR i = 0 TO loaded_vocab_size - 1
                GET #file, , mask_value
                IF mask_value = 0 THEN
                    tokenizer.output_allowed(i) = 0
                ELSE
                    tokenizer.output_allowed(i) = 1
                END IF
            NEXT i
        ELSEIF extension_marker = TOKENIZER_MODE_MAGIC THEN
            IF mode_seen <> 0 THEN
                error_message = "duplicate tokenizer mode extension"
                GOTO load_failed
            END IF
            mode_seen = 1
            GET #file, , mode_code
            IF mode_code < TOKENIZER_MODE_BYTE OR mode_code > TOKENIZER_MODE_LEXICON THEN
                error_message = "unknown tokenizer mode code"
                GOTO load_failed
            END IF
            tokenizer.tokenizer_mode = mode_code
        ELSE
            error_message = "unknown vocabulary extension marker"
            GOTO load_failed
        END IF
    WEND

    IF tokenizer.output_allowed(EOT_TOKEN) = 0 THEN
        error_message = "EOT token masked in output mask"
        GOTO load_failed
    END IF
    IF tokenizer.output_allowed(UNK_TOKEN) <> 0 THEN
        error_message = "UNK token allowed in output mask"
        GOTO load_failed
    END IF
    FOR i = 0 TO 255
        IF i >= 32 AND i <= 126 THEN
            IF tokenizer.output_allowed(i + 2) = 0 THEN
                error_message = "printable byte masked in output mask"
                GOTO load_failed
            END IF
        ELSE
            IF tokenizer.output_allowed(i + 2) <> 0 THEN
                error_message = "non-printable byte allowed in output mask"
                GOTO load_failed
            END IF
        END IF
    NEXT i

    IF mode_seen = 0 THEN
        IF loaded_merge_count > 0 THEN
            tokenizer.tokenizer_mode = TOKENIZER_MODE_BPE
        ELSE
            tokenizer.tokenizer_mode = TOKENIZER_MODE_BYTE
        END IF
    END IF

    IF tokenizer.tokenizer_mode = TOKENIZER_MODE_BYTE THEN
        IF loaded_vocab_size <> BYTE_VOCAB_SIZE OR loaded_merge_count <> 0 THEN
            error_message = "byte tokenizer mode has extra vocabulary or merges"
            GOTO load_failed
        END IF
    ELSEIF tokenizer.tokenizer_mode = TOKENIZER_MODE_BPE THEN
        IF loaded_merge_count <= 0 THEN
            error_message = "BPE tokenizer mode has no merges"
            GOTO load_failed
        END IF
    ELSEIF tokenizer.tokenizer_mode = TOKENIZER_MODE_LEXICON THEN
        IF loaded_merge_count <> 0 THEN
            error_message = "lexicon tokenizer mode cannot contain merges"
            GOTO load_failed
        END IF
        IF loaded_vocab_size <= BYTE_VOCAB_SIZE THEN
            error_message = "lexicon tokenizer mode has no lexicon entries"
            GOTO load_failed
        END IF
    ELSE
        error_message = "unknown tokenizer mode"
        GOTO load_failed
    END IF

    IF tokenizer.tokenizer_mode = TOKENIZER_MODE_BPE THEN
        tokenizer.use_bpe = 1
    ELSE
        tokenizer.use_bpe = 0
    END IF
    IF tokenizer.tokenizer_mode = TOKENIZER_MODE_LEXICON THEN
        TokenizerBuildLexiconBuckets tokenizer
    END IF

    CLOSE file

    PRINT "Loaded vocabulary with "; tokenizer.vocab_size; " tokens and "; _
          tokenizer.merge_count; " merges, mode "; tokenizer.tokenizer_mode
    EXIT SUB

load_failed:
    PRINT "ERROR: Invalid vocabulary file "; filename; ": "; error_message
    CLOSE file
    ERROR 5
END SUB

' Save vocabulary to a file
SUB SaveVocabulary(tokenizer AS Tokenizer, filename AS STRING)
    DIM file AS LONG, i AS INTEGER
    DIM output_mask_magic_value AS LONG
    DIM tokenizer_mode_magic_value AS LONG

    ' Open vocabulary file
    file = FREEFILE
    OPEN filename FOR BINARY AS file

    ' Write the vocabulary size
    PUT #file, , tokenizer.vocab_size

    ' Write tokens
    FOR i = 0 TO tokenizer.vocab_size - 1
        ' Write token length
        PUT #file, , tokenizer.vocab(i).token_len

        ' Write token string (fixed size)
        PUT #file, , tokenizer.vocab(i).token

        ' Write token ID
        PUT #file, , tokenizer.vocab(i).token_id
    NEXT i

    ' Write merge operations
    PUT #file, , tokenizer.merge_count
    FOR i = 0 TO tokenizer.merge_count - 1
        PUT #file, , tokenizer.merges(i).first
        PUT #file, , tokenizer.merges(i).second
        PUT #file, , tokenizer.merges(i).result
        PUT #file, , tokenizer.merges(i).priority
    NEXT i

    output_mask_magic_value = OUTPUT_MASK_MAGIC
    PUT #file, , output_mask_magic_value
    PUT #file, , tokenizer.vocab_size
    FOR i = 0 TO tokenizer.vocab_size - 1
        PUT #file, , tokenizer.output_allowed(i)
    NEXT i

    tokenizer_mode_magic_value = TOKENIZER_MODE_MAGIC
    PUT #file, , tokenizer_mode_magic_value
    PUT #file, , tokenizer.tokenizer_mode

    CLOSE file

    PRINT "Saved vocabulary with "; tokenizer.vocab_size; " tokens and "; _
          tokenizer.merge_count; " merges, mode "; tokenizer.tokenizer_mode
END SUB

' Add a merge operation to the tokenizer
SUB AddMergeOperation(BYREF tokenizer AS Tokenizer, first AS INTEGER, second AS INTEGER, result AS INTEGER, priority AS INTEGER)
    ' Check if we have room
    IF tokenizer.merge_count >= MAX_MERGES THEN
        PRINT "ERROR: Merge table is full"
        RETURN
    END IF

    ' Add merge operation
    tokenizer.merges(tokenizer.merge_count).first = first
    tokenizer.merges(tokenizer.merge_count).second = second
    tokenizer.merges(tokenizer.merge_count).result = result
    tokenizer.merges(tokenizer.merge_count).priority = priority

    ' Increment merge count
    tokenizer.merge_count = tokenizer.merge_count + 1
    tokenizer.tokenizer_mode = TOKENIZER_MODE_BPE
    tokenizer.use_bpe = 1
END SUB

' *******************************************************
' * Tokenization Functions                              *
' *******************************************************

' Convert bytes to tokens (byte-level tokenization)
SUB BytesToTokens(tokenizer AS Tokenizer, bytes() AS BYTE, byte_count AS INTEGER, tokens() AS INTEGER, BYREF token_count AS INTEGER)
    DIM i AS INTEGER

    ' Resize tokens array
    REDIM tokens(0 TO byte_count)

    ' For byte-level tokenization, each byte becomes a token
    token_count = byte_count
    FOR i = 0 TO byte_count - 1
        ' Token ID is byte value + 2 (offsets for special tokens)
        tokens(i) = bytes(i) + 2
    NEXT i

    ' Add end-of-text token
    tokens(token_count) = EOT_TOKEN
    token_count = token_count + 1
END SUB

' Convert a string to tokens (including BPE if enabled)
SUB StringToTokens(tokenizer AS Tokenizer, input_text AS STRING, tokens() AS INTEGER, BYREF token_count AS INTEGER)
    DIM bytes() AS BYTE
    DIM byte_count AS INTEGER
    DIM i AS INTEGER
    DIM clean_text AS STRING

    ' Convert string to bytes
    clean_text = CleanTokenizerText(input_text)
    byte_count = LEN(clean_text)
    IF byte_count <= 0 THEN
        REDIM tokens(0 TO 0)
        tokens(0) = EOT_TOKEN
        token_count = 1
        EXIT SUB
    END IF
    REDIM bytes(0 TO byte_count - 1)

    FOR i = 1 TO byte_count
        bytes(i - 1) = ASC(MID$(clean_text, i, 1))
    NEXT i

    ' Tokenize bytes
    IF tokenizer.tokenizer_mode = TOKENIZER_MODE_BPE THEN
        ' Use BPE tokenization
        BPETokenize(tokenizer, bytes(), byte_count, tokens(), token_count)
    ELSEIF tokenizer.tokenizer_mode = TOKENIZER_MODE_LEXICON THEN
        ' Use longest-match lexicon tokenization
        LexiconTokenize(tokenizer, bytes(), byte_count, tokens(), token_count)
    ELSE
        ' Use byte-level tokenization
        BytesToTokens(tokenizer, bytes(), byte_count, tokens(), token_count)
    END IF
END SUB

' Return true when one vocabulary piece matches bytes at the given byte offset.
FUNCTION TokenPieceMatchesBytes(tokenizer AS Tokenizer, vocab_idx AS INTEGER, bytes() AS BYTE, start_idx AS INTEGER, byte_count AS INTEGER) AS INTEGER
    DIM piece_len AS INTEGER
    DIM i AS INTEGER

    IF vocab_idx < 0 OR vocab_idx >= tokenizer.vocab_size THEN RETURN 0
    piece_len = tokenizer.vocab(vocab_idx).token_len
    IF piece_len <= 0 THEN RETURN 0
    IF start_idx + piece_len > byte_count THEN RETURN 0

    FOR i = 0 TO piece_len - 1
        IF ASC(MID$(tokenizer.vocab(vocab_idx).token, i + 1, 1)) <> bytes(start_idx + i) THEN
            RETURN 0
        END IF
    NEXT i

    RETURN 1
END FUNCTION

' Return true for bytes that can continue a tokenizer word. Lexicon pieces that
' end in one of these bytes cannot match when the next source byte also
' continues a word, because that would split a longer word into a prefix piece
' plus alphabetic bytes.
FUNCTION TokenizerLexiconWordByte(byte_value AS INTEGER) AS INTEGER
    IF byte_value >= ASC("0") AND byte_value <= ASC("9") THEN RETURN 1
    IF byte_value >= ASC("A") AND byte_value <= ASC("Z") THEN RETURN 1
    IF byte_value >= ASC("a") AND byte_value <= ASC("z") THEN RETURN 1
    IF byte_value = ASC(".") OR byte_value = ASC("+") OR byte_value = ASC("/") OR _
       byte_value = ASC("_") OR byte_value = ASC("-") THEN RETURN 1
    RETURN 0
END FUNCTION

FUNCTION TokenizerLexiconBoundaryNextOK(bytes() AS BYTE, next_idx AS INTEGER, byte_count AS INTEGER) AS INTEGER
    DIM next_byte AS INTEGER
    DIM following_byte AS INTEGER

    IF next_idx >= byte_count THEN RETURN 1
    next_byte = bytes(next_idx)
    IF next_byte = ASC(".") THEN
        IF next_idx + 1 >= byte_count THEN RETURN 1
        following_byte = bytes(next_idx + 1)
        IF TokenizerLexiconWordByte(following_byte) = 0 THEN RETURN 1
        RETURN 0
    END IF
    IF TokenizerLexiconWordByte(next_byte) <> 0 THEN RETURN 0
    RETURN 1
END FUNCTION

' Return true when a matching lexicon piece ends at a word boundary.
FUNCTION TokenPieceBoundaryMatches(tokenizer AS Tokenizer, vocab_idx AS INTEGER, bytes() AS BYTE, start_idx AS INTEGER, byte_count AS INTEGER) AS INTEGER
    DIM piece_len AS INTEGER
    DIM last_byte AS INTEGER

    IF vocab_idx < 0 OR vocab_idx >= tokenizer.vocab_size THEN RETURN 0
    piece_len = tokenizer.vocab(vocab_idx).token_len
    IF piece_len <= 0 THEN RETURN 0
    IF start_idx + piece_len >= byte_count THEN RETURN 1

    last_byte = ASC(MID$(tokenizer.vocab(vocab_idx).token, piece_len, 1))
    IF TokenizerLexiconWordByte(last_byte) <> 0 AND _
       TokenizerLexiconBoundaryNextOK(bytes(), start_idx + piece_len, byte_count) = 0 THEN RETURN 0
    RETURN 1
END FUNCTION

' Build first-byte metadata for lexicon pieces. The tokenizer can then try only
' token lengths that exist for the current byte and use the exact hash lookup,
' longest length first. Buckets are retained for diagnostics and future trie
' experiments, but generation uses the length table to avoid scanning thousands
' of space-prefixed tokens.
SUB TokenizerBuildLexiconBuckets(BYREF tokenizer AS Tokenizer)
    DIM i AS INTEGER
    DIM b AS INTEGER
    DIM order_idx AS INTEGER
    DIM start_idx AS INTEGER
    DIM end_idx AS INTEGER
    DIM scan_idx AS INTEGER
    DIM best_pos AS INTEGER
    DIM tmp_idx AS INTEGER
    DIM vocab_idx AS INTEGER
    DIM best_vocab_idx AS INTEGER
    DIM token_len AS INTEGER
    DIM best_len AS INTEGER
    DIM token_id AS INTEGER
    DIM best_token_id AS INTEGER

    FOR b = 0 TO 255
        tokenizer.lexicon_bucket_start(b) = 0
        tokenizer.lexicon_bucket_count(b) = 0
        FOR i = 1 TO MAX_TOKEN_LENGTH
            tokenizer.lexicon_length_present(b, i) = 0
        NEXT i
    NEXT b
    FOR i = 0 TO MAX_VOCAB_SIZE - 1
        tokenizer.lexicon_order(i) = -1
    NEXT i

    FOR i = 0 TO tokenizer.vocab_size - 1
        IF tokenizer.vocab(i).token_id >= BYTE_VOCAB_SIZE AND tokenizer.vocab(i).token_len > 0 THEN
            b = ASC(LEFT$(tokenizer.vocab(i).token, 1))
            tokenizer.lexicon_bucket_count(b) = tokenizer.lexicon_bucket_count(b) + 1
            IF tokenizer.vocab(i).token_len >= 1 AND tokenizer.vocab(i).token_len <= MAX_TOKEN_LENGTH THEN
                tokenizer.lexicon_length_present(b, tokenizer.vocab(i).token_len) = 1
            END IF
        END IF
    NEXT i

    order_idx = 0
    FOR b = 0 TO 255
        tokenizer.lexicon_bucket_start(b) = order_idx
        order_idx = order_idx + tokenizer.lexicon_bucket_count(b)
        tokenizer.lexicon_bucket_count(b) = 0
    NEXT b

    FOR i = 0 TO tokenizer.vocab_size - 1
        IF tokenizer.vocab(i).token_id >= BYTE_VOCAB_SIZE AND tokenizer.vocab(i).token_len > 0 THEN
            b = ASC(LEFT$(tokenizer.vocab(i).token, 1))
            order_idx = tokenizer.lexicon_bucket_start(b) + tokenizer.lexicon_bucket_count(b)
            tokenizer.lexicon_order(order_idx) = i
            tokenizer.lexicon_bucket_count(b) = tokenizer.lexicon_bucket_count(b) + 1
        END IF
    NEXT i

    FOR b = 0 TO 255
        start_idx = tokenizer.lexicon_bucket_start(b)
        end_idx = start_idx + tokenizer.lexicon_bucket_count(b) - 1
        IF end_idx > start_idx THEN
            FOR i = start_idx TO end_idx - 1
                best_pos = i
                best_vocab_idx = tokenizer.lexicon_order(i)
                best_len = tokenizer.vocab(best_vocab_idx).token_len
                best_token_id = tokenizer.vocab(best_vocab_idx).token_id
                FOR scan_idx = i + 1 TO end_idx
                    vocab_idx = tokenizer.lexicon_order(scan_idx)
                    token_len = tokenizer.vocab(vocab_idx).token_len
                    token_id = tokenizer.vocab(vocab_idx).token_id
                    IF token_len > best_len OR (token_len = best_len AND token_id < best_token_id) THEN
                        best_pos = scan_idx
                        best_vocab_idx = vocab_idx
                        best_len = token_len
                        best_token_id = token_id
                    END IF
                NEXT scan_idx
                IF best_pos <> i THEN
                    tmp_idx = tokenizer.lexicon_order(i)
                    tokenizer.lexicon_order(i) = tokenizer.lexicon_order(best_pos)
                    tokenizer.lexicon_order(best_pos) = tmp_idx
                END IF
            NEXT i
        END IF
    NEXT b
END SUB

' Longest-match lexicon tokenization. Lexicon entries are complete words,
' phrases, or punctuation chunks; bytes remain the fallback for all text.
SUB LexiconTokenize(tokenizer AS Tokenizer, bytes() AS BYTE, byte_count AS INTEGER, tokens() AS INTEGER, BYREF token_count AS INTEGER)
    DIM idx AS INTEGER
    DIM best_idx AS INTEGER
    DIM best_len AS INTEGER
    DIM piece_len AS INTEGER
    DIM token_id AS INTEGER
    DIM first_byte AS INTEGER
    DIM max_len AS INTEGER
    DIM candidate AS STRING
    DIM k AS INTEGER

    REDIM tokens(0 TO byte_count)
    token_count = 0
    idx = 0

    WHILE idx < byte_count
        best_idx = -1
        best_len = 0

        first_byte = bytes(idx)
        max_len = tokenizer.max_token_length
        IF max_len > byte_count - idx THEN max_len = byte_count - idx

        FOR piece_len = max_len TO 1 STEP -1
            IF tokenizer.lexicon_length_present(first_byte, piece_len) <> 0 THEN
                candidate = ""
                FOR k = 0 TO piece_len - 1
                    candidate = candidate + CHR$(bytes(idx + k))
                NEXT k

                best_idx = FindToken(tokenizer, candidate)
                IF best_idx >= 0 THEN
                    token_id = tokenizer.vocab(best_idx).token_id
                    IF token_id >= BYTE_VOCAB_SIZE AND _
                       TokenPieceBoundaryMatches(tokenizer, best_idx, bytes(), idx, byte_count) <> 0 THEN
                        best_len = piece_len
                        EXIT FOR
                    ELSE
                        best_idx = -1
                    END IF
                END IF
            END IF
        NEXT piece_len

        IF best_idx >= 0 THEN
            tokens(token_count) = tokenizer.vocab(best_idx).token_id
            token_count = token_count + 1
            idx = idx + best_len
        ELSE
            tokens(token_count) = bytes(idx) + 2
            token_count = token_count + 1
            idx = idx + 1
        END IF
    WEND

    tokens(token_count) = EOT_TOKEN
    token_count = token_count + 1
END SUB

' Ranked BPE tokenization algorithm. This mirrors the host tokenizer: each pass
' finds the adjacent token pair with the lowest merge priority, then merges all
' occurrences of that pair before scanning again.
SUB BPETokenize(tokenizer AS Tokenizer, bytes() AS BYTE, byte_count AS INTEGER, tokens() AS INTEGER, BYREF token_count AS INTEGER)
    DIM byte_tokens() AS INTEGER
    DIM byte_token_count AS INTEGER
    DIM i AS INTEGER, j AS INTEGER, k AS INTEGER
    DIM merged AS INTEGER
    DIM best_found AS INTEGER
    DIM best_priority AS INTEGER
    DIM best_first AS INTEGER
    DIM best_second AS INTEGER
    DIM best_result AS INTEGER

    ' Start with byte-level tokens
    BytesToTokens(tokenizer, bytes(), byte_count, byte_tokens(), byte_token_count)

    ' Store working copy of tokens
    DIM working_tokens() AS INTEGER
    REDIM working_tokens(0 TO byte_token_count - 1)
    token_count = byte_token_count
    FOR i = 0 TO byte_token_count - 1
        working_tokens(i) = byte_tokens(i)
    NEXT i

    ' Apply merges repeatedly
    DO
        merged = 0
        best_found = 0
        best_priority = &H7FFFFFFF
        best_first = -1
        best_second = -1
        best_result = -1

        ' Find the highest-priority merge currently present anywhere.
        FOR i = 0 TO token_count - 2 ' Check pairs of tokens
            DIM first AS INTEGER, second AS INTEGER
            first = working_tokens(i)
            second = working_tokens(i + 1)

            FOR j = 0 TO tokenizer.merge_count - 1
                IF tokenizer.merges(j).first = first AND tokenizer.merges(j).second = second THEN
                    IF tokenizer.merges(j).priority < best_priority THEN
                        best_priority = tokenizer.merges(j).priority
                        best_first = first
                        best_second = second
                        best_result = tokenizer.merges(j).result
                        best_found = 1
                    END IF
                END IF
            NEXT j
        NEXT i

        IF best_found <> 0 THEN
            ' Merge every non-overlapping occurrence of the selected pair.
            i = 0
            k = 0
            WHILE i < token_count
                IF i + 1 < token_count AND working_tokens(i) = best_first AND _
                   working_tokens(i + 1) = best_second THEN
                    working_tokens(k) = best_result
                    i = i + 2
                    k = k + 1
                    merged = 1
                ELSE
                    working_tokens(k) = working_tokens(i)
                    i = i + 1
                    k = k + 1
                END IF
            WEND
            IF merged <> 0 THEN
                token_count = k
            END IF
        END IF
    LOOP WHILE merged = 1

    ' Copy final tokens to output
    REDIM tokens(0 TO token_count - 1)
    FOR i = 0 TO token_count - 1
        tokens(i) = working_tokens(i)
    NEXT i
END SUB

' Convert tokens back to a string
FUNCTION TokensToString(tokenizer AS Tokenizer, tokens() AS INTEGER, token_count AS INTEGER) AS STRING
    DIM result AS STRING
    DIM i AS INTEGER, j AS INTEGER, token_idx AS INTEGER

    result = ""

    FOR i = 0 TO token_count - 1
        ' Skip special tokens
        IF tokens(i) = EOT_TOKEN THEN
            ' End of text, stop decoding
            EXIT FOR
        END IF

        ' Find token in vocabulary
        token_idx = -1
        FOR j = 0 TO tokenizer.vocab_size - 1
            IF tokenizer.vocab(j).token_id = tokens(i) THEN
                token_idx = j
                EXIT FOR
            END IF
        NEXT j

        ' Add token to result if found
        IF token_idx >= 0 THEN
            result = result + LEFT$(tokenizer.vocab(token_idx).token, tokenizer.vocab(token_idx).token_len)
        ELSE
            ' Unknown token, use the unknown token placeholder
            result = result + "<|unk|>"
        END IF
    NEXT i

    RETURN result
END FUNCTION

' Return whether a token may be sampled as output. BPE merge intermediates can
' still be valid for prompt/context encoding while being masked during sampling.
FUNCTION TokenizerOutputAllowed(tokenizer AS Tokenizer, token_id AS INTEGER) AS INTEGER
    IF token_id < 0 OR token_id >= tokenizer.vocab_size THEN RETURN 0
    IF tokenizer.output_allowed(token_id) = 0 THEN RETURN 0
    RETURN 1
END FUNCTION

' *******************************************************
' * BPE Training (for building vocabulary)              *
' *******************************************************

' Simplified BPE training from a corpus
SUB TrainBPE(BYREF tokenizer AS Tokenizer, corpus_filename AS STRING, max_merge_count AS INTEGER)
    DIM file AS LONG
    DIM i AS INTEGER, j AS INTEGER, merge_count AS INTEGER
    DIM input_line AS STRING
    DIM bytes() AS BYTE
    DIM tokens() AS INTEGER
    DIM token_count AS INTEGER

    ' Bounded sparse pair counts. A dense MAX_VOCAB_SIZE^2 table is too large for DOS.
    CONST PAIR_TABLE_SIZE = 8192
    DIM pair_keys(0 TO PAIR_TABLE_SIZE - 1) AS LONG
    DIM pair_counts(0 TO PAIR_TABLE_SIZE - 1) AS LONG
    DIM pair_slot AS INTEGER
    DIM pair_idx AS LONG
    DIM pair_key AS LONG
    DIM best_pair_first AS INTEGER, best_pair_second AS INTEGER
    DIM best_slot AS INTEGER

    FOR i = 0 TO PAIR_TABLE_SIZE - 1
        pair_keys(i) = -1
        pair_counts(i) = 0
    NEXT i

    ' Open corpus file
    file = FREEFILE
    OPEN corpus_filename FOR INPUT AS file

    ' Disable BPE during training
    tokenizer.tokenizer_mode = TOKENIZER_MODE_BYTE
    tokenizer.use_bpe = 0

    ' Read all lines and count token pairs
    PRINT "Counting token pairs..."
    WHILE EOF(file) = 0
        LINE INPUT #file, input_line

        ' Convert to tokens
        StringToTokens(tokenizer, input_line, tokens(), token_count)

        ' Count pairs
        FOR i = 0 TO token_count - 2
            pair_key = tokens(i) * MAX_VOCAB_SIZE + tokens(i + 1)
            pair_slot = pair_key MOD PAIR_TABLE_SIZE

            WHILE pair_keys(pair_slot) <> -1 AND pair_keys(pair_slot) <> pair_key
                pair_slot = pair_slot + 1
                IF pair_slot >= PAIR_TABLE_SIZE THEN pair_slot = 0
            WEND

            IF pair_keys(pair_slot) = -1 THEN
                pair_keys(pair_slot) = pair_key
                pair_counts(pair_slot) = 1
            ELSE
                pair_counts(pair_slot) = pair_counts(pair_slot) + 1
            END IF
        NEXT i
    WEND

    CLOSE file

    ' Now perform merges
    PRINT "Performing BPE merges..."
    merge_count = 0

    WHILE merge_count < max_merge_count
        ' Find most frequent pair
        DIM best_count AS LONG, best_pair AS LONG
        best_count = 0
        best_pair = 0
        best_slot = -1

        FOR i = 0 TO PAIR_TABLE_SIZE - 1
            IF pair_keys(i) <> -1 AND pair_counts(i) > best_count THEN
                best_count = pair_counts(i)
                best_pair = pair_keys(i)
                best_pair_first = best_pair \ MAX_VOCAB_SIZE
                best_pair_second = best_pair MOD MAX_VOCAB_SIZE
                best_slot = i
            END IF
        NEXT i

        ' If no pairs left to merge, stop
        IF best_count = 0 THEN
            EXIT WHILE
        END IF

        ' Create new token for this merge
        DIM new_token AS STRING
        DIM token_str1 AS STRING, token_str2 AS STRING
        DIM idx1 AS INTEGER, idx2 AS INTEGER

        ' Find the tokens in vocabulary by ID
        idx1 = -1
        idx2 = -1
        FOR i = 0 TO tokenizer.vocab_size - 1
            IF tokenizer.vocab(i).token_id = best_pair_first THEN
                idx1 = i
            END IF
            IF tokenizer.vocab(i).token_id = best_pair_second THEN
                idx2 = i
            END IF
        NEXT i

        ' Skip if we can't find the tokens
        IF idx1 < 0 OR idx2 < 0 THEN
            ' Clear this pair count to avoid reselecting it
            IF best_slot >= 0 THEN pair_counts(best_slot) = 0
            CONTINUE WHILE
        END IF

        ' Combine tokens
        token_str1 = LEFT$(tokenizer.vocab(idx1).token, tokenizer.vocab(idx1).token_len)
        token_str2 = LEFT$(tokenizer.vocab(idx2).token, tokenizer.vocab(idx2).token_len)
        new_token = token_str1 + token_str2

        ' Check if combined token is too long
        IF LEN(new_token) > MAX_TOKEN_LENGTH THEN
            ' Skip this merge
            IF best_slot >= 0 THEN pair_counts(best_slot) = 0
            CONTINUE WHILE
        END IF

        ' Add new token to vocabulary
        DIM new_token_id AS INTEGER, new_token_idx AS INTEGER
        new_token_id = tokenizer.vocab_size + 2 ' +2 to avoid special tokens
        new_token_idx = AddToken(tokenizer, new_token, new_token_id)

        ' Add merge operation
        AddMergeOperation(tokenizer, best_pair_first, best_pair_second, new_token_id, merge_count)

        ' Clear pair count to avoid selecting it again
        IF best_slot >= 0 THEN pair_counts(best_slot) = 0

        ' Increment merge count
        merge_count = merge_count + 1

        ' Print progress every 100 merges
        IF merge_count MOD 100 = 0 THEN
            PRINT "Completed "; merge_count; " merges"
        END IF
    WEND

    ' Re-enable BPE
    tokenizer.tokenizer_mode = TOKENIZER_MODE_BPE
    tokenizer.use_bpe = 1

    PRINT "BPE training complete. Added "; merge_count; " merges."
END SUB

' *******************************************************
' * Main Interface Functions                            *
' *******************************************************

' Initialize the default tokenizer
SUB InitializeDefaultTokenizer()
    InitTokenizer(g_tokenizer)
END SUB

' Encode a string to token IDs
SUB Encode(input_text AS STRING, tokens() AS INTEGER, BYREF token_count AS INTEGER)
    StringToTokens(g_tokenizer, input_text, tokens(), token_count)
END SUB

' Decode token IDs to a string
FUNCTION Decode(tokens() AS INTEGER, token_count AS INTEGER) AS STRING
    RETURN TokensToString(g_tokenizer, tokens(), token_count)
END FUNCTION

' Load the default vocabulary from a file
SUB LoadDefaultVocabulary(filename AS STRING)
    LoadVocabulary(g_tokenizer, filename)
END SUB

' Save the default vocabulary to a file
SUB SaveDefaultVocabulary(filename AS STRING)
    SaveVocabulary(g_tokenizer, filename)
END SUB

' *******************************************************
' * Testing Functions                                   *
' *******************************************************

' Test tokenization
SUB TestTokenization()
    DIM tokens() AS INTEGER
    DIM token_count AS INTEGER
    DIM decoded AS STRING
    DIM i AS INTEGER

    ' Initialize tokenizer
    InitializeDefaultTokenizer()

    ' Test strings
    DIM test_strings(0 TO 4) AS STRING
    test_strings(0) = "Hello, world!"
    test_strings(1) = "This is a test of the tokenizer."
    test_strings(2) = "GPT-2 BASIC for 486 computers."
    test_strings(3) = "Artificial intelligence on vintage hardware."
    test_strings(4) = "The quick brown fox jumps over the lazy dog."

    ' Test each string
    FOR i = 0 TO 4
        PRINT "Testing: '"; test_strings(i); "'"

        ' Encode
        Encode(test_strings(i), tokens(), token_count)

        ' Print tokens
        PRINT "  Token count: "; token_count
        PRINT "  Tokens: ";

        DIM j AS INTEGER
        FOR j = 0 TO MIN(token_count - 1, 20)
            PRINT tokens(j);
            IF j < MIN(token_count - 1, 20) THEN
                PRINT ", ";
            END IF
        NEXT j

        IF token_count > 20 THEN
            PRINT "..."
        ELSE
            PRINT
        END IF

        ' Decode
        decoded = Decode(tokens(), token_count)
        PRINT "  Decoded: '"; decoded; "'"

        ' Check if decoding matched the original
        IF decoded = test_strings(i) THEN
            PRINT "  Test PASSED: Decoded text matches original"
        ELSE
            PRINT "  Test FAILED: Decoded text differs from original"
        END IF

        PRINT
    NEXT i
END SUB

' Test simple BPE training
SUB TestBPETraining()
    DIM file AS LONG
    DIM i AS INTEGER

    ' Initialize tokenizer
    InitializeDefaultTokenizer()

    ' Create a test corpus
    file = FREEFILE
    OPEN "test_corpus.txt" FOR OUTPUT AS file
    PRINT #file, "Hello world! This is a test corpus for BPE training."
    PRINT #file, "The goal is to see if we can learn simple merges."
    PRINT #file, "Words like 'the', 'and', 'ing' should be merged."
    PRINT #file, "Let's repeat some patterns to make them more frequent:"
    PRINT #file, "the the the the the"
    PRINT #file, "and and and and and"
    PRINT #file, "ing ing ing ing ing"
    PRINT #file, "test test test test test"
    CLOSE file

    ' Train BPE on the corpus
    PRINT "Training BPE on test corpus..."
    TrainBPE(g_tokenizer, "test_corpus.txt", 20) ' Just do a few merges for testing

    ' Print the vocabulary after training
    PRINT "Vocabulary after training:"
    FOR i = 0 TO MIN(g_tokenizer.vocab_size - 1, 50)
        PRINT i; ": '"; LEFT$(g_tokenizer.vocab(i).token, g_tokenizer.vocab(i).token_len); _
              "' (ID: "; g_tokenizer.vocab(i).token_id; ")"
    NEXT i

    ' Print the merges
    PRINT "Merge operations:"
    FOR i = 0 TO MIN(g_tokenizer.merge_count - 1, 20)
        PRINT i; ": "; g_tokenizer.merges(i).first; " + "; _
              g_tokenizer.merges(i).second; " -> "; _
              g_tokenizer.merges(i).result; " (priority: "; _
              g_tokenizer.merges(i).priority; ")"
    NEXT i

    ' Clean up
    KILL "test_corpus.txt"
END SUB

' Main tokenizer test routine
SUB TestTokenizer()
    PRINT "Testing Tokenizer Module"
    PRINT "========================"
    PRINT

    ' Test basic tokenization
    TestTokenization()
    PRINT

    ' Test BPE training
    TestBPETraining()
END SUB
