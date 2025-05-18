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
    vocab_hash(0 TO 4095) AS INTEGER ' Simple hash table for token lookup
    use_bpe AS INTEGER ' Whether to use BPE or just byte-level tokenization
    max_token_length AS INTEGER
    cache_enabled AS INTEGER
END TYPE

' Global tokenizer instance
DIM SHARED g_tokenizer AS Tokenizer

' *******************************************************
' * Initialization and Setup                            *
' *******************************************************

' Initialize a tokenizer
SUB InitTokenizer(BYREF tokenizer AS Tokenizer)
    DIM i AS INTEGER
    
    ' Clear tokenizer data
    tokenizer.vocab_size = 0
    tokenizer.merge_count = 0
    tokenizer.use_bpe = 1 ' Enable BPE by default
    tokenizer.max_token_length = MAX_TOKEN_LENGTH
    tokenizer.cache_enabled = 1 ' Enable cache by default
    
    ' Initialize hash table to -1 (empty)
    FOR i = 0 TO 4095
        tokenizer.vocab_hash(i) = -1
    NEXT i
    
    ' Initialize with basic tokens (special tokens and ASCII chars)
    ' Add special tokens
    AddToken(tokenizer, "<|endoftext|>", EOT_TOKEN)
    AddToken(tokenizer, "<|unk|>", UNK_TOKEN)
    
    ' Add basic ASCII characters (0-255)
    FOR i = 0 TO 255
        DIM char_token AS STRING * 2
        char_token = CHR$(i)
        AddToken(tokenizer, char_token, i + 2) ' +2 because 0,1 are special
    NEXT i
    
    PRINT "Initialized tokenizer with "; tokenizer.vocab_size; " tokens"
END SUB

' Add a token to the vocabulary
FUNCTION AddToken(BYREF tokenizer AS Tokenizer, token AS STRING, token_id AS INTEGER) AS INTEGER
    DIM i AS INTEGER, hash_val AS INTEGER
    
    ' Check if we have room
    IF tokenizer.vocab_size >= MAX_VOCAB_SIZE THEN
        PRINT "ERROR: Vocabulary is full"
        RETURN -1
    END IF
    
    ' Check if token already exists
    hash_val = HashToken(token) MOD 4096
    i = tokenizer.vocab_hash(hash_val)
    
    WHILE i >= 0
        IF LEFT$(tokenizer.vocab(i).token, tokenizer.vocab(i).token_len) = token THEN
            ' Token already exists
            RETURN i
        END IF
        ' Linear probing for collision
        hash_val = (hash_val + 1) MOD 4096
        i = tokenizer.vocab_hash(hash_val)
    WEND
    
    ' Add new token
    i = tokenizer.vocab_size
    tokenizer.vocab(i).token = token
    tokenizer.vocab(i).token_len = LEN(token)
    tokenizer.vocab(i).token_id = token_id
    tokenizer.vocab(i).count = 0
    
    ' Update hash table
    hash_val = HashToken(token) MOD 4096
    WHILE tokenizer.vocab_hash(hash_val) >= 0
        ' Linear probing for collision
        hash_val = (hash_val + 1) MOD 4096
    WEND
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
    DIM hash_val AS INTEGER, i AS INTEGER
    
    hash_val = HashToken(token) MOD 4096
    i = tokenizer.vocab_hash(hash_val)
    
    WHILE i >= 0
        IF LEFT$(tokenizer.vocab(i).token, tokenizer.vocab(i).token_len) = token THEN
            RETURN i
        END IF
        ' Linear probing for collision
        hash_val = (hash_val + 1) MOD 4096
        i = tokenizer.vocab_hash(hash_val)
    WEND
    
    ' Token not found
    RETURN -1
END FUNCTION

' *******************************************************
' * Vocabulary Management                               *
' *******************************************************

' Load vocabulary from a file
SUB LoadVocabulary(BYREF tokenizer AS Tokenizer, filename AS STRING)
    DIM file AS LONG, i AS INTEGER
    DIM token AS STRING * MAX_TOKEN_LENGTH
    DIM token_len AS INTEGER
    DIM token_id AS INTEGER
    
    ' Open vocabulary file
    file = FREEFILE
    OPEN filename FOR BINARY AS file
    
    ' Read the vocabulary size
    GET #file, , tokenizer.vocab_size
    
    ' Read token_size tokens
    FOR i = 0 TO tokenizer.vocab_size - 1
        ' Read token length
        GET #file, , token_len
        
        ' Read token string (fixed size)
        GET #file, , token
        
        ' Read token ID
        GET #file, , token_id
        
        ' Store in vocabulary
        tokenizer.vocab(i).token = token
        tokenizer.vocab(i).token_len = token_len
        tokenizer.vocab(i).token_id = token_id
        tokenizer.vocab(i).count = 0
        
        ' Update hash table
        DIM hash_val AS INTEGER
        hash_val = HashToken(LEFT$(token, token_len)) MOD 4096
        WHILE tokenizer.vocab_hash(hash_val) >= 0
            ' Linear probing for collision
            hash_val = (hash_val + 1) MOD 4096
        WEND
        tokenizer.vocab_hash(hash_val) = i
    NEXT i
    
    ' Read merge operations
    GET #file, , tokenizer.merge_count
    FOR i = 0 TO tokenizer.merge_count - 1
        GET #file, , tokenizer.merges(i).first
        GET #file, , tokenizer.merges(i).second
        GET #file, , tokenizer.merges(i).result
        GET #file, , tokenizer.merges(i).priority
    NEXT i
    
    CLOSE file
    
    PRINT "Loaded vocabulary with "; tokenizer.vocab_size; " tokens and "; _
          tokenizer.merge_count; " merges"
END SUB

' Save vocabulary to a file
SUB SaveVocabulary(tokenizer AS Tokenizer, filename AS STRING)
    DIM file AS LONG, i AS INTEGER
    
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
    
    CLOSE file
    
    PRINT "Saved vocabulary with "; tokenizer.vocab_size; " tokens and "; _
          tokenizer.merge_count; " merges"
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
END SUB

' *******************************************************
' * Tokenization Functions                              *
' *******************************************************

' Convert bytes to tokens (byte-level tokenization)
SUB BytesToTokens(tokenizer AS Tokenizer, bytes() AS BYTE, byte_count AS INTEGER, BYREF tokens() AS INTEGER, BYREF token_count AS INTEGER)
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
SUB StringToTokens(tokenizer AS Tokenizer, input_text AS STRING, BYREF tokens() AS INTEGER, BYREF token_count AS INTEGER)
    DIM bytes() AS BYTE
    DIM byte_count AS INTEGER
    DIM i AS INTEGER
    
    ' Convert string to bytes
    byte_count = LEN(input_text)
    REDIM bytes(0 TO byte_count - 1)
    
    FOR i = 1 TO byte_count
        bytes(i - 1) = ASC(MID$(input_text, i, 1))
    NEXT i
    
    ' Tokenize bytes
    IF tokenizer.use_bpe THEN
        ' Use BPE tokenization
        BPETokenize(tokenizer, bytes(), byte_count, tokens(), token_count)
    ELSE
        ' Use byte-level tokenization
        BytesToTokens(tokenizer, bytes(), byte_count, tokens(), token_count)
    END IF
END SUB

' Simplified BPE tokenization algorithm
SUB BPETokenize(tokenizer AS Tokenizer, bytes() AS BYTE, byte_count AS INTEGER, BYREF tokens() AS INTEGER, BYREF token_count AS INTEGER)
    DIM byte_tokens() AS INTEGER
    DIM byte_token_count AS INTEGER
    DIM i AS INTEGER, j AS INTEGER, k AS INTEGER
    DIM merged AS INTEGER
    
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
        
        ' Find and apply best merge
        FOR i = 0 TO token_count - 2 ' Check pairs of tokens
            DIM first AS INTEGER, second AS INTEGER
            first = working_tokens(i)
            second = working_tokens(i + 1)
            
            ' Look for matching merge operation
            DIM best_merge AS INTEGER, best_priority AS INTEGER
            best_merge = -1
            best_priority = &H7FFFFFFF ' max int
            
            FOR j = 0 TO tokenizer.merge_count - 1
                IF tokenizer.merges(j).first = first AND tokenizer.merges(j).second = second THEN
                    ' Found a potential merge
                    IF tokenizer.merges(j).priority < best_priority THEN
                        best_merge = j
                        best_priority = tokenizer.merges(j).priority
                    END IF
                END IF
            NEXT j
            
            ' Apply best merge if found
            IF best_merge >= 0 THEN
                ' Apply the merge
                working_tokens(i) = tokenizer.merges(best_merge).result
                
                ' Shift remaining tokens
                FOR k = i + 1 TO token_count - 2
                    working_tokens(k) = working_tokens(k + 1)
                NEXT k
                
                ' Decrease token count
                token_count = token_count - 1
                
                ' Mark as merged
                merged = 1
                
                ' Only apply one merge per iteration (for simplicity)
                EXIT FOR
            END IF
        NEXT i
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
    DIM i AS INTEGER, token_idx AS INTEGER
    
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

' *******************************************************
' * BPE Training (for building vocabulary)              *
' *******************************************************

' Simplified BPE training from a corpus
SUB TrainBPE(BYREF tokenizer AS Tokenizer, corpus_filename AS STRING, max_merges AS INTEGER)
    DIM file AS LONG
    DIM i AS INTEGER, j AS INTEGER, merge_count AS INTEGER
    DIM line AS STRING
    DIM bytes() AS BYTE
    DIM tokens() AS INTEGER
    DIM token_count AS INTEGER
    
    ' Initialize pair counts
    DIM pair_count AS LONG
    DIM pairs(0 TO MAX_VOCAB_SIZE * MAX_VOCAB_SIZE - 1) AS LONG ' Simple for demonstration
    DIM best_pair_first AS INTEGER, best_pair_second AS INTEGER
    
    ' Open corpus file
    file = FREEFILE
    OPEN corpus_filename FOR INPUT AS file
    
    ' Disable BPE during training
    tokenizer.use_bpe = 0
    
    ' Read all lines and count token pairs
    PRINT "Counting token pairs..."
    WHILE NOT EOF(file)
        LINE INPUT #file, line
        
        ' Convert to tokens
        StringToTokens(tokenizer, line, tokens(), token_count)
        
        ' Count pairs
        FOR i = 0 TO token_count - 2
            DIM pair_idx AS LONG
            pair_idx = tokens(i) * MAX_VOCAB_SIZE + tokens(i + 1)
            IF pair_idx >= 0 AND pair_idx < MAX_VOCAB_SIZE * MAX_VOCAB_SIZE THEN
                pairs(pair_idx) = pairs(pair_idx) + 1
            END IF
        NEXT i
    WEND
    
    CLOSE file
    
    ' Now perform merges
    PRINT "Performing BPE merges..."
    merge_count = 0
    
    WHILE merge_count < max_merges
        ' Find most frequent pair
        DIM best_count AS LONG, best_pair AS LONG
        best_count = 0
        best_pair = 0
        
        FOR i = 0 TO MAX_VOCAB_SIZE - 1
            FOR j = 0 TO MAX_VOCAB_SIZE - 1
                DIM pair_idx AS LONG
                pair_idx = i * MAX_VOCAB_SIZE + j
                
                IF pairs(pair_idx) > best_count THEN
                    best_count = pairs(pair_idx)
                    best_pair = pair_idx
                    best_pair_first = i
                    best_pair_second = j
                END IF
            NEXT j
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
            pairs(best_pair) = 0
            CONTINUE WHILE
        END IF
        
        ' Combine tokens
        token_str1 = LEFT$(tokenizer.vocab(idx1).token, tokenizer.vocab(idx1).token_len)
        token_str2 = LEFT$(tokenizer.vocab(idx2).token, tokenizer.vocab(idx2).token_len)
        new_token = token_str1 + token_str2
        
        ' Check if combined token is too long
        IF LEN(new_token) > MAX_TOKEN_LENGTH THEN
            ' Skip this merge
            pairs(best_pair) = 0
            CONTINUE WHILE
        END IF
        
        ' Add new token to vocabulary
        DIM new_token_id AS INTEGER, new_token_idx AS INTEGER
        new_token_id = tokenizer.vocab_size + 2 ' +2 to avoid special tokens
        new_token_idx = AddToken(tokenizer, new_token, new_token_id)
        
        ' Add merge operation
        AddMergeOperation(tokenizer, best_pair_first, best_pair_second, new_token_id, merge_count)
        
        ' Clear pair count to avoid selecting it again
        pairs(best_pair) = 0
        
        ' Increment merge count
        merge_count = merge_count + 1
        
        ' Print progress every 100 merges
        IF merge_count MOD 100 = 0 THEN
            PRINT "Completed "; merge_count; " merges"
        END IF
    WEND
    
    ' Re-enable BPE
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
SUB Encode(input_text AS STRING, BYREF tokens() AS INTEGER, BYREF token_count AS INTEGER)
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
