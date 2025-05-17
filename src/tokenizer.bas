' Tokenizer implementation for the GPT-2-like model.
' This file provides functions for vocabulary management and text tokenization,
' optimized for 486-era constraints.

' Include necessary files
#INCLUDE "data_structures.bas"
#INCLUDE "file_io.bas" ' For vocabulary loading

' Constants
CONST MAX_TOKEN_LENGTH AS INTEGER = 32 ' Maximum length of a token in characters
CONST MAX_VOCAB_SIZE AS INTEGER = 5000 ' Maximum vocabulary size (5K target)
CONST BPE_MERGES_SIZE AS INTEGER = 10000 ' Maximum number of BPE merges
CONST SPECIAL_TOKENS_COUNT AS INTEGER = 5 ' Number of special tokens (PAD, UNK, BOS, EOS, etc.)
CONST BUFFER_SIZE AS INTEGER = 1024 ' Buffer size for tokenization

' Special token constants
CONST PAD_TOKEN_ID AS INTEGER = 0
CONST UNK_TOKEN_ID AS INTEGER = 1
CONST BOS_TOKEN_ID AS INTEGER = 2
CONST EOS_TOKEN_ID AS INTEGER = 3
CONST MASK_TOKEN_ID AS INTEGER = 4

' Type to represent a vocabulary token
TYPE VocabToken
    text AS STRING * MAX_TOKEN_LENGTH
    id AS INTEGER
    frequency AS INTEGER ' For potential sorting by frequency
END TYPE

' Type to represent a merge rule for BPE (Byte Pair Encoding)
TYPE BPEMerge
    first AS STRING * MAX_TOKEN_LENGTH
    second AS STRING * MAX_TOKEN_LENGTH
    result AS STRING * MAX_TOKEN_LENGTH
    priority AS INTEGER ' Lower number = higher priority
END TYPE

' Global vocabulary and BPE merge rules
DIM Vocab(MAX_VOCAB_SIZE - 1) AS VocabToken
DIM BPEMerges(BPE_MERGES_SIZE - 1) AS BPEMerge
DIM VocabSize AS INTEGER
DIM BPEMergesCount AS INTEGER

' Flag to track if tokenizer is initialized
DIM TokenizerInitialized AS INTEGER

' Type for token buffer used during tokenization
TYPE TokenBuffer
    tokens(BUFFER_SIZE - 1) AS INTEGER
    count AS INTEGER
END TYPE

' Initialize the tokenizer with default special tokens
SUB InitTokenizer()
    IF TokenizerInitialized = 1 THEN
        PRINT "Tokenizer already initialized."
        EXIT SUB
    END IF
    
    ' Initialize default special tokens
    Vocab(PAD_TOKEN_ID).text = "<PAD>"
    Vocab(PAD_TOKEN_ID).id = PAD_TOKEN_ID
    
    Vocab(UNK_TOKEN_ID).text = "<UNK>"
    Vocab(UNK_TOKEN_ID).id = UNK_TOKEN_ID
    
    Vocab(BOS_TOKEN_ID).text = "<BOS>"
    Vocab(BOS_TOKEN_ID).id = BOS_TOKEN_ID
    
    Vocab(EOS_TOKEN_ID).text = "<EOS>"
    Vocab(EOS_TOKEN_ID).id = EOS_TOKEN_ID
    
    Vocab(MASK_TOKEN_ID).text = "<MASK>"
    Vocab(MASK_TOKEN_ID).id = MASK_TOKEN_ID
    
    VocabSize = SPECIAL_TOKENS_COUNT
    BPEMergesCount = 0
    
    TokenizerInitialized = 1
    
    PRINT "Tokenizer initialized with default special tokens."
END SUB

' Load vocabulary from a file
' Format: One token per line, with format "token [tab] id"
FUNCTION LoadVocabulary(filepath AS STRING) AS INTEGER
    DIM file_num AS INTEGER
    DIM line_buffer AS STRING * 512
    DIM token_text AS STRING
    DIM token_id AS INTEGER
    DIM token_count AS INTEGER
    
    IF TokenizerInitialized = 0 THEN
        InitTokenizer()
    END IF
    
    ' Open the vocabulary file
    file_num = FREEFILE
    
    ON ERROR GOTO LoadError
    OPEN filepath FOR INPUT AS #file_num
    ON ERROR GOTO 0
    
    token_count = SPECIAL_TOKENS_COUNT ' Start after special tokens
    
    ' Read each line
    WHILE NOT EOF(file_num) AND token_count < MAX_VOCAB_SIZE
        LINE INPUT #file_num, line_buffer
        
        ' Skip empty lines
        IF LEN(TRIM$(line_buffer)) = 0 THEN
            CONTINUE WHILE
        END IF
        
        ' Parse line: "token [tab] id"
        DIM tab_pos AS INTEGER
        tab_pos = INSTR(line_buffer, CHR$(9)) ' Tab character
        
        IF tab_pos > 0 THEN
            token_text = LEFT$(line_buffer, tab_pos - 1)
            token_id = VAL(MID$(line_buffer, tab_pos + 1))
        ELSE
            ' If no tab, assume just the token and assign sequential ID
            token_text = TRIM$(line_buffer)
            token_id = token_count
        END IF
        
        ' Make sure token isn't too long
        IF LEN(token_text) > MAX_TOKEN_LENGTH THEN
            token_text = LEFT$(token_text, MAX_TOKEN_LENGTH)
        END IF
        
        ' Store in vocab
        Vocab(token_count).text = token_text
        Vocab(token_count).id = token_id
        
        token_count = token_count + 1
    WEND
    
    CLOSE #file_num
    
    VocabSize = token_count
    PRINT "Loaded "; token_count; " tokens into vocabulary."
    
    FUNCTION = token_count
    EXIT FUNCTION
    
LoadError:
    PRINT "Error loading vocabulary from "; filepath
    CLOSE #file_num
    FUNCTION = 0
END FUNCTION

' Load BPE merge rules from a file
' Format: "first second result priority"
FUNCTION LoadBPEMerges(filepath AS STRING) AS INTEGER
    DIM file_num AS INTEGER
    DIM line_buffer AS STRING * 512
    DIM first AS STRING
    DIM second AS STRING
    DIM result AS STRING
    DIM priority AS INTEGER
    DIM merge_count AS INTEGER
    
    ' Open the BPE merges file
    file_num = FREEFILE
    
    ON ERROR GOTO LoadError
    OPEN filepath FOR INPUT AS #file_num
    ON ERROR GOTO 0
    
    merge_count = 0
    
    ' Read each line
    WHILE NOT EOF(file_num) AND merge_count < BPE_MERGES_SIZE
        LINE INPUT #file_num, line_buffer
        
        ' Skip empty lines
        IF LEN(TRIM$(line_buffer)) = 0 THEN
            CONTINUE WHILE
        END IF
        
        ' Parse line: "first second result priority"
        DIM tokens(3) AS STRING
        DIM token_count AS INTEGER
        DIM current_pos AS INTEGER
        DIM token_start AS INTEGER
        
        token_count = 0
        current_pos = 1
        
        ' Simple tokenizer for the merge rule line
        WHILE current_pos <= LEN(line_buffer) AND token_count < 4
            ' Skip whitespace
            WHILE current_pos <= LEN(line_buffer) AND INSTR(" " + CHR$(9), MID$(line_buffer, current_pos, 1)) > 0
                current_pos = current_pos + 1
            WEND
            
            ' Break if end of line
            IF current_pos > LEN(line_buffer) THEN
                EXIT WHILE
            END IF
            
            ' Find token end
            token_start = current_pos
            WHILE current_pos <= LEN(line_buffer) AND INSTR(" " + CHR$(9), MID$(line_buffer, current_pos, 1)) = 0
                current_pos = current_pos + 1
            WEND
            
            ' Extract token
            tokens(token_count) = MID$(line_buffer, token_start, current_pos - token_start)
            token_count = token_count + 1
        WEND
        
        ' Validate we have all required parts
        IF token_count >= 3 THEN
            first = tokens(0)
            second = tokens(1)
            result = tokens(2)
            
            ' Priority is optional, default to merge_count (order in file)
            IF token_count >= 4 THEN
                priority = VAL(tokens(3))
            ELSE
                priority = merge_count
            END IF
            
            ' Store in BPEMerges
            BPEMerges(merge_count).first = first
            BPEMerges(merge_count).second = second
            BPEMerges(merge_count).result = result
            BPEMerges(merge_count).priority = priority
            
            merge_count = merge_count + 1
        END IF
    WEND
    
    CLOSE #file_num
    
    BPEMergesCount = merge_count
    PRINT "Loaded "; merge_count; " BPE merge rules."
    
    FUNCTION = merge_count
    EXIT FUNCTION
    
LoadError:
    PRINT "Error loading BPE merges from "; filepath
    CLOSE #file_num
    FUNCTION = 0
END FUNCTION

' Convert token text to token ID
FUNCTION GetTokenID(token_text AS STRING) AS INTEGER
    DIM i AS INTEGER
    
    ' Linear search for token (could be optimized with a hash map or binary search)
    FOR i = 0 TO VocabSize - 1
        IF RTRIM$(Vocab(i).text) = token_text THEN
            FUNCTION = Vocab(i).id
            EXIT FUNCTION
        END IF
    NEXT i
    
    ' Not found, return UNK token
    FUNCTION = UNK_TOKEN_ID
END FUNCTION

' Convert token ID to token text
FUNCTION GetTokenText(token_id AS INTEGER) AS STRING
    DIM i AS INTEGER
    
    ' Linear search for token ID (could be optimized with direct lookup if IDs are sequential)
    FOR i = 0 TO VocabSize - 1
        IF Vocab(i).id = token_id THEN
            FUNCTION = RTRIM$(Vocab(i).text)
            EXIT FUNCTION
        END IF
    NEXT i
    
    ' Not found, return UNK token text
    FUNCTION = RTRIM$(Vocab(UNK_TOKEN_ID).text)
END FUNCTION

' Simple character-level tokenization (fallback)
' This is used when BPE is not available or for unknown tokens
SUB CharTokenize(text AS STRING, buffer AS TokenBuffer)
    DIM i AS INTEGER
    DIM char AS STRING * 1
    DIM token_id AS INTEGER
    
    buffer.count = 0
    
    ' Process each character
    FOR i = 1 TO LEN(text)
        IF buffer.count >= BUFFER_SIZE THEN
            EXIT SUB ' Buffer full
        END IF
        
        char = MID$(text, i, 1)
        
        ' Try to find character in vocabulary
        token_id = GetTokenID(char)
        
        ' Add to buffer
        buffer.tokens(buffer.count) = token_id
        buffer.count = buffer.count + 1
    NEXT i
END SUB

' Apply BPE merge rules iteratively until no more merges can be applied
' This is a simplified version of BPE tokenization
SUB ApplyBPEMerges(tokens() AS STRING, count AS INTEGER, BYREF merged_count AS INTEGER)
    DIM merged(BUFFER_SIZE - 1) AS STRING
    DIM changed AS INTEGER
    DIM i AS INTEGER, j AS INTEGER
    
    ' Copy tokens to merged array
    FOR i = 0 TO count - 1
        merged(i) = tokens(i)
    NEXT i
    merged_count = count
    
    ' Apply merges until no more changes
    DO
        changed = 0
        
        i = 0
        WHILE i < merged_count - 1
            ' Try to find a merge rule for this pair
            FOR j = 0 TO BPEMergesCount - 1
                IF merged(i) = RTRIM$(BPEMerges(j).first) AND merged(i + 1) = RTRIM$(BPEMerges(j).second) THEN
                    ' Apply merge: replace first token with merged token, remove second token
                    merged(i) = RTRIM$(BPEMerges(j).result)
                    
                    ' Shift remaining tokens
                    FOR j = i + 1 TO merged_count - 2
                        merged(j) = merged(j + 1)
                    NEXT j
                    
                    merged_count = merged_count - 1
                    changed = 1
                    
                    ' Don't increment i, so we can check for further merges with the new token
                    EXIT FOR
                END IF
            NEXT j
            
            ' If no merge was applied, move to next position
            IF changed = 0 THEN
                i = i + 1
            END IF
        WEND
    LOOP WHILE changed = 1
    
    ' Copy result back to tokens array
    FOR i = 0 TO merged_count - 1
        tokens(i) = merged(i)
    NEXT i
END SUB

' Simple space-based tokenization with BPE merges
SUB TokenizeWithBPE(text AS STRING, buffer AS TokenBuffer)
    DIM word_buffer AS STRING * 256
    DIM tokens(BUFFER_SIZE - 1) AS STRING
    DIM token_count AS INTEGER
    DIM merged_count AS INTEGER
    DIM i AS INTEGER, j AS INTEGER
    DIM char AS STRING * 1
    DIM in_word AS INTEGER
    
    buffer.count = 0
    token_count = 0
    in_word = 0
    
    ' First pass: split into words
    FOR i = 1 TO LEN(text)
        char = MID$(text, i, 1)
        
        ' Check if character is a word character or whitespace/punctuation
        IF INSTR("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", char) > 0 THEN
            ' Word character
            IF in_word = 0 THEN
                ' Starting a new word
                in_word = 1
                word_buffer = ""
            END IF
            
            word_buffer = word_buffer + char
        ELSE
            ' Whitespace or punctuation
            IF in_word = 1 THEN
                ' End of a word, add it to tokens
                IF token_count < BUFFER_SIZE THEN
                    tokens(token_count) = RTRIM$(word_buffer)
                    token_count = token_count + 1
                END IF
                in_word = 0
            END IF
            
            ' Add the punctuation/whitespace as its own token
            IF token_count < BUFFER_SIZE THEN
                tokens(token_count) = char
                token_count = token_count + 1
            END IF
        END IF
    NEXT i
    
    ' If we ended in a word, add it
    IF in_word = 1 AND token_count < BUFFER_SIZE THEN
        tokens(token_count) = RTRIM$(word_buffer)
        token_count = token_count + 1
    END IF
    
    ' Apply BPE merges if available
    IF BPEMergesCount > 0 THEN
        ApplyBPEMerges tokens(), token_count, merged_count
        token_count = merged_count
    END IF
    
    ' Convert tokens to token IDs
    FOR i = 0 TO token_count - 1
        IF buffer.count >= BUFFER_SIZE THEN
            EXIT SUB ' Buffer full
        END IF
        
        buffer.tokens(buffer.count) = GetTokenID(tokens(i))
        buffer.count = buffer.count + 1
    NEXT i
END SUB

' Main tokenization function that chooses the appropriate tokenization method
' and handles special tokens
FUNCTION Tokenize(text AS STRING, max_length AS INTEGER) AS INTEGER()
    DIM buffer AS TokenBuffer
    DIM result(max_length - 1) AS INTEGER
    DIM context_length AS INTEGER
    DIM i AS INTEGER
    
    ' Initialize result with padding tokens
    FOR i = 0 TO max_length - 1
        result(i) = PAD_TOKEN_ID
    NEXT i
    
    ' Verify tokenizer is initialized
    IF TokenizerInitialized = 0 THEN
        InitTokenizer()
    END IF
    
    ' Add BOS token at the beginning
    result(0) = BOS_TOKEN_ID
    context_length = 1
    
    ' Choose tokenization method based on available resources
    IF BPEMergesCount > 0 THEN
        ' Use BPE tokenization
        TokenizeWithBPE text, buffer
    ELSE
        ' Use character-level tokenization
        CharTokenize text, buffer
    END IF
    
    ' Copy tokens to result, up to max_length - 2 (leave room for EOS)
    FOR i = 0 TO buffer.count - 1
        IF context_length >= max_length - 1 THEN
            EXIT FOR ' Leave room for EOS
        END IF
        
        result(context_length) = buffer.tokens(i)
        context_length = context_length + 1
    NEXT i
    
    ' Add EOS token at the end
    IF context_length < max_length THEN
        result(context_length) = EOS_TOKEN_ID
        context_length = context_length + 1
    END IF
    
    FUNCTION = result
END FUNCTION

' Create a more realistic simple vocabulary for testing
SUB CreateSimpleVocabulary(filepath AS STRING)
    DIM file_num AS INTEGER
    DIM i AS INTEGER
    
    file_num = FREEFILE
    
    ON ERROR GOTO CreateError
    OPEN filepath FOR OUTPUT AS #file_num
    ON ERROR GOTO 0
    
    ' First write special tokens
    PRINT #file_num, "<PAD>" + CHR$(9) + "0"
    PRINT #file_num, "<UNK>" + CHR$(9) + "1"
    PRINT #file_num, "<BOS>" + CHR$(9) + "2"
    PRINT #file_num, "<EOS>" + CHR$(9) + "3"
    PRINT #file_num, "<MASK>" + CHR$(9) + "4"
    
    ' Common English words
    DIM common_words(95) AS STRING
    common_words(0) = "the"
    common_words(1) = "of"
    common_words(2) = "and"
    common_words(3) = "to"
    common_words(4) = "in"
    common_words(5) = "a"
    common_words(6) = "is"
    common_words(7) = "that"
    common_words(8) = "for"
    common_words(9) = "it"
    common_words(10) = "as"
    common_words(11) = "was"
    common_words(12) = "with"
    common_words(13) = "be"
    common_words(14) = "by"
    common_words(15) = "on"
    common_words(16) = "not"
    common_words(17) = "he"
    common_words(18) = "I"
    common_words(19) = "this"
    common_words(20) = "are"
    common_words(21) = "or"
    common_words(22) = "his"
    common_words(23) = "from"
    common_words(24) = "at"
    common_words(25) = "which"
    common_words(26) = "but"
    common_words(27) = "have"
    common_words(28) = "an"
    common_words(29) = "had"
    common_words(30) = "they"
    common_words(31) = "you"
    common_words(32) = "were"
    common_words(33) = "there"
    common_words(34) = "one"
    common_words(35) = "all"
    common_words(36) = "we"
    common_words(37) = "can"
    common_words(38) = "her"
    common_words(39) = "has"
    common_words(40) = "their"
    common_words(41) = "been"
    common_words(42) = "if"
    common_words(43) = "more"
    common_words(44) = "when"
    common_words(45) = "will"
    common_words(46) = "would"
    common_words(47) = "who"
    common_words(48) = "so"
    common_words(49) = "no"
    common_words(50) = "she"
    common_words(51) = "how"
    common_words(52) = "me"
    common_words(53) = "what"
    common_words(54) = "up"
    common_words(55) = "out"
    common_words(56) = "do"
    common_words(57) = "time"
    common_words(58) = "other"
    common_words(59) = "my"
    common_words(60) = "into"
    common_words(61) = "than"
    common_words(62) = "its"
    common_words(63) = "only"
    common_words(64) = "some"
    common_words(65) = "could"
    common_words(66) = "new"
    common_words(67) = "them"
    common_words(68) = "man"
    common_words(69) = "about"
    common_words(70) = "then"
    common_words(71) = "first"
    common_words(72) = "also"
    common_words(73) = "after"
    common_words(74) = "any"
    common_words(75) = "like"
    common_words(76) = "should"
    common_words(77) = "people"
    common_words(78) = "now"
    common_words(79) = "these"
    common_words(80) = "may"
    common_words(81) = "such"
    common_words(82) = "your"
    common_words(83) = "over"
    common_words(84) = "most"
    common_words(85) = "through"
    common_words(86) = "between"
    common_words(87) = "before"
    common_words(88) = "very"
    common_words(89) = "many"
    common_words(90) = "just"
    common_words(91) = "those"
    common_words(92) = "where"
    common_words(93) = "here"
    common_words(94) = "must"
    common_words(95) = "way"
    
    ' Write common words
    FOR i = 0 TO 95
        PRINT #file_num, common_words(i) + CHR$(9) + STR$(i + SPECIAL_TOKENS_COUNT)
    NEXT i
    
    ' Add ASCII characters
    FOR i = 32 TO 126
        PRINT #file_num, CHR$(i) + CHR$(9) + STR$(i + SPECIAL_TOKENS_COUNT + 96)
    NEXT i
    
    CLOSE #file_num
    PRINT "Created simple vocabulary with "; SPECIAL_TOKENS_COUNT + 96 + 95; " tokens at "; filepath
    EXIT SUB
    
CreateError:
    PRINT "Error creating vocabulary file: "; filepath
    CLOSE #file_num
END SUB
