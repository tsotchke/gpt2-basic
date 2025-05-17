' Main program file for the GPT-2-like model in BASIC.
' This file initializes the model and starts text generation.

' Include necessary files
#INCLUDE "data_structures.bas"
#INCLUDE "quantization.bas"
#INCLUDE "matrix_ops.bas"
#INCLUDE "transformer_components.bas"
#INCLUDE "softmax_fixed.bas"
#INCLUDE "block_sparse.bas"
#INCLUDE "file_io.bas"
#INCLUDE "tokenizer.bas"
#INCLUDE "model.bas"

' Constants for program operation
CONST DEFAULT_PROMPT AS STRING = "The quick brown fox jumps over the lazy dog."
CONST DEFAULT_MAX_TOKENS AS INTEGER = 20
CONST MODEL_DIR AS STRING = "model_data/"
CONST ERR_FILE_NOT_FOUND AS INTEGER = 53

' --- Main Program Entry Point ---
SUB Main()
    DIM prompt_text AS STRING
    DIM max_generated_tokens AS INTEGER
    
    ' Display header
    PRINT "GPT-2 in BASIC (486 Compatible)"
    PRINT "-------------------------------"
    
    ' Initialize random seed for sampling
    RANDOMIZE TIMER
    
    ' Parse command line arguments (if any)
    prompt_text = DEFAULT_PROMPT
    max_generated_tokens = DEFAULT_MAX_TOKENS
    
    ' Check if directories exist, create if needed
    IF DirExists(MODEL_DIR) = 0 THEN
        PRINT "Creating model data directory..."
        MkDir MODEL_DIR
    END IF
    
    ' First-time setup: Create vocabulary for testing if it doesn't exist
    IF FileExists(MODEL_DIR + "vocabulary.txt") = 0 THEN
        PRINT "Creating sample vocabulary for testing..."
        InitTokenizer()
        CreateSimpleVocabulary(MODEL_DIR + "vocabulary.txt")
    END IF
    
    ' Ask for a custom prompt if the user wants
    PRINT "Default prompt: "; prompt_text
    PRINT "Enter custom prompt (or press Enter to use default):"
    
    DIM input_line AS STRING
    LINE INPUT input_line
    
    IF LEN(TRIM$(input_line)) > 0 THEN
        prompt_text = input_line
    END IF
    
    ' Ask for number of tokens to generate
    PRINT "Default number of tokens to generate: "; max_generated_tokens
    PRINT "Enter custom number (or press Enter to use default):"
    
    LINE INPUT input_line
    
    IF LEN(TRIM$(input_line)) > 0 THEN
        max_generated_tokens = VAL(input_line)
        IF max_generated_tokens <= 0 THEN
            max_generated_tokens = DEFAULT_MAX_TOKENS
        END IF
    END IF
    
    ' Display generation settings
    PRINT ""
    PRINT "Text Generation Settings:"
    PRINT "  Prompt: "; prompt_text
    PRINT "  Tokens to generate: "; max_generated_tokens
    PRINT "  Model directory: "; MODEL_DIR
    PRINT ""
    
    ' Try to load vocabulary if available
    PRINT "Loading vocabulary..."
    InitTokenizer()
    ON ERROR GOTO LoadVocabError
    DIM vocab_size AS INTEGER = LoadVocabulary(MODEL_DIR + "vocabulary.txt")
    ON ERROR GOTO 0
    
    IF vocab_size > 0 THEN
        PRINT "Loaded "; vocab_size; " tokens in vocabulary."
    ELSE
        PRINT "Using basic vocabulary."
    END IF
    
    ' Perform text generation
    PRINT ""
    PRINT "Starting text generation..."
    PRINT "---"
    PRINT ""
    
    ON ERROR GOTO GenerationError
    GenerateText prompt_text, max_generated_tokens
    ON ERROR GOTO 0
    
    PRINT ""
    PRINT "---"
    PRINT "Text generation complete."
    
    ' Clean exit
    EXIT SUB
    
LoadVocabError:
    IF ERR = ERR_FILE_NOT_FOUND THEN
        PRINT "Vocabulary file not found. Using default vocabulary."
        RESUME NEXT
    ELSE
        PRINT "Error loading vocabulary: "; ERR
        RESUME NEXT
    END IF
    
GenerationError:
    PRINT ""
    PRINT "Error during text generation: "; ERR
    PRINT "Exiting..."
    EXIT SUB
END SUB

' Check if a directory exists
FUNCTION DirExists(dirpath AS STRING) AS INTEGER
    DIM attrib AS INTEGER
    
    ON ERROR GOTO DirError
    attrib = _FILEATTR(dirpath, 1)
    ON ERROR GOTO 0
    
    ' If we get here, the directory exists
    FUNCTION = 1
    EXIT FUNCTION
    
DirError:
    FUNCTION = 0
END FUNCTION

' Check if a file exists
FUNCTION FileExists(filepath AS STRING) AS INTEGER
    DIM file_num AS INTEGER
    
    file_num = FREEFILE
    
    ON ERROR GOTO FileError
    OPEN filepath FOR INPUT AS #file_num
    CLOSE #file_num
    ON ERROR GOTO 0
    
    ' If we get here, the file exists
    FUNCTION = 1
    EXIT FUNCTION
    
FileError:
    FUNCTION = 0
END FUNCTION

' Create directory with error handling
SUB MkDir(dirpath AS STRING)
    ON ERROR GOTO MkDirError
    MKDIR dirpath
    ON ERROR GOTO 0
    EXIT SUB
    
MkDirError:
    PRINT "Error creating directory "; dirpath; ": "; ERR
END SUB

' Run the main program
Main
END
