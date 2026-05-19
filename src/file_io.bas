' File I/O implementation for the GPT-2-like model.
' This file contains functions for streaming model parameters from disk,
' allowing the model to work within the 32MB RAM constraints of a 486.

' Include necessary files
#INCLUDE "data_structures.bas"
#INCLUDE "quantization.bas"

' Constants
CONST MAX_FILENAME_LENGTH AS INTEGER = 256
CONST CHUNK_SIZE AS INTEGER = 8192 ' Read chunk size in bytes (8KB is a reasonable size for 486 disk I/O)
CONST HEADER_SIZE AS INTEGER = 12 ' Size of the matrix header in bytes (rows, cols, data_type)

' Structure to track file handles for matrix streaming
TYPE FileHandle
    file_num AS INTEGER      ' File number used in BASIC
    is_open AS INTEGER       ' Flag to track if file is open (1 = open, 0 = closed)
    filename AS STRING * MAX_FILENAME_LENGTH  ' The filename being accessed
    total_bytes AS LONG      ' Total file size in bytes
    current_pos AS LONG      ' Current position in the file
END TYPE

' Structure for model parameters stored on disk
TYPE ModelFileInfo
    token_embed_file AS FileHandle   ' Token embedding weights
    pos_embed_file AS FileHandle     ' Positional embedding weights
    layer_files(16) AS FileHandle    ' Files for each layer (adjust size as needed)
    output_file AS FileHandle        ' Output layer weights
    vocab_file AS FileHandle         ' Vocabulary file
    config_file AS FileHandle        ' Model configuration parameters
    num_layers AS INTEGER            ' Number of layers (extracted from config)
END TYPE

' Initialize a file handle
SUB InitFileHandle(handle AS FileHandle, filename AS STRING)
    handle.filename = filename
    handle.file_num = FREEFILE ' Get next available file number
    handle.is_open = 0         ' File is not open yet
    handle.total_bytes = 0
    handle.current_pos = 0
END SUB

' Open a file for binary reading
FUNCTION OpenReadFile(handle AS FileHandle) AS INTEGER
    IF handle.is_open = 1 THEN
        RETURN 1 ' File already open
    END IF
    
    ' Open the file in binary mode
    ON ERROR GOTO OpenError
    OPEN handle.filename FOR BINARY AS #handle.file_num
    ON ERROR GOTO 0
    
    handle.is_open = 1
    handle.total_bytes = LOF(handle.file_num) ' Length of file
    handle.current_pos = 0
    
    RETURN 1
    
OpenError:
    PRINT "Error opening file: "; handle.filename
    RETURN 0
END FUNCTION

' Close a file
SUB CloseFile(handle AS FileHandle)
    IF handle.is_open = 1 THEN
        CLOSE #handle.file_num
        handle.is_open = 0
    END IF
END SUB

FUNCTION ModelDirectoryHasGPT2BasicCheckpoint(base_path AS STRING) AS INTEGER
    IF DIR(base_path + "\GPT2CFG.TXT") <> "" THEN
        IF DIR(base_path + "\GPT2FX.BIN") <> "" AND DIR(base_path + "\GPT2EXP.BIN") <> "" THEN RETURN 1
        IF DIR(base_path + "\GPT2WT.BIN") <> "" THEN RETURN 1
    END IF

    IF DIR(base_path + "\TINYCFG.TXT") <> "" THEN
        IF DIR(base_path + "\TINYFX.BIN") <> "" AND DIR(base_path + "\TINYEXP.BIN") <> "" THEN RETURN 1
        IF DIR(base_path + "\TINYWT.BIN") <> "" THEN RETURN 1
    END IF

    RETURN 0
END FUNCTION

SUB InitGPT2BasicModelFiles(model AS ModelFileInfo, base_path AS STRING)
    DIM cfg_name AS STRING
    DIM fixed_name AS STRING
    DIM exp_name AS STRING
    DIM float_name AS STRING
    DIM profile_name AS STRING
    DIM file_num AS INTEGER
    DIM line_buffer AS STRING
    DIM eq_pos AS INTEGER
    DIM key_text AS STRING
    DIM value_text AS STRING

    cfg_name = "GPT2CFG.TXT"
    fixed_name = "GPT2FX.BIN"
    exp_name = "GPT2EXP.BIN"
    float_name = "GPT2WT.BIN"
    profile_name = "PROFILE.TXT"

    IF DIR(base_path + "\" + cfg_name) = "" THEN cfg_name = "TINYCFG.TXT"
    IF DIR(base_path + "\" + fixed_name) = "" THEN fixed_name = "TINYFX.BIN"
    IF DIR(base_path + "\" + exp_name) = "" THEN exp_name = "TINYEXP.BIN"
    IF DIR(base_path + "\" + float_name) = "" THEN float_name = "TINYWT.BIN"

    InitFileHandle model.token_embed_file, base_path + "\" + fixed_name
    InitFileHandle model.pos_embed_file, base_path + "\" + exp_name
    InitFileHandle model.output_file, base_path + "\" + float_name
    InitFileHandle model.vocab_file, base_path + "\" + profile_name
    InitFileHandle model.config_file, base_path + "\" + cfg_name

    model.num_layers = 1
    file_num = FREEFILE
    ON ERROR GOTO cfg_done
    OPEN base_path + "\" + cfg_name FOR INPUT AS #file_num

    WHILE EOF(file_num) = 0
        LINE INPUT #file_num, line_buffer
        line_buffer = TRIM$(line_buffer)
        IF line_buffer <> "" AND LEFT$(line_buffer, 1) <> "#" THEN
            eq_pos = INSTR(line_buffer, "=")
            IF eq_pos > 0 THEN
                key_text = LCASE$(TRIM$(LEFT$(line_buffer, eq_pos - 1)))
                value_text = TRIM$(MID$(line_buffer, eq_pos + 1))
                IF key_text = "n_layer" THEN model.num_layers = VAL(value_text)
            END IF
        END IF
    WEND
    CLOSE #file_num

cfg_done:
    ON ERROR GOTO 0
    IF model.num_layers < 1 THEN model.num_layers = 1
END SUB

' Initialize the model file information
SUB InitModelFiles(model AS ModelFileInfo, base_path AS STRING)
    DIM layer_idx AS INTEGER

    IF ModelDirectoryHasGPT2BasicCheckpoint(base_path) <> 0 THEN
        InitGPT2BasicModelFiles model, base_path
        RETURN
    END IF
    
    ' Initialize file handles with proper paths
    InitFileHandle model.token_embed_file, base_path + "\TOKEMB.BIN"
    InitFileHandle model.pos_embed_file, base_path + "\POSEMB.BIN"
    InitFileHandle model.output_file, base_path + "\OUTPUT.BIN"
    InitFileHandle model.vocab_file, base_path + "\VOCAB.TXT"
    InitFileHandle model.config_file, base_path + "\CONFIG.BIN"
    
    ' Read configuration file to get number of layers
    model.num_layers = 2 ' Default value if config file can't be read
    
    ' Try to read the number of layers from config file
    IF OpenReadFile(model.config_file) THEN
        ' Format of config file:
        ' - First 4 bytes: EMBEDDING_DIM (integer)
        ' - Next 4 bytes: NUM_HEADS (integer)
        ' - Next 4 bytes: NUM_LAYERS (integer)
        ' - Next 4 bytes: CONTEXT_LENGTH (integer)
        ' - Next 4 bytes: VOCAB_SIZE (integer)
        DIM buffer(4) AS INTEGER
        
        ' Seek to the NUM_LAYERS position (8 bytes from start)
        SEEK #model.config_file.file_num, 9 ' 1-based indexing in BASIC
        
        ' Read the value
        GET #model.config_file.file_num, , buffer(0)
        model.num_layers = buffer(0)
        
        CloseFile model.config_file
    END IF
    
    ' Initialize layer files
    FOR layer_idx = 0 TO model.num_layers - 1
        InitFileHandle model.layer_files(layer_idx), base_path + "\L" + LTRIM$(STR$(layer_idx)) + ".BIN"
    NEXT layer_idx
END SUB

' Clean up all file handles
SUB CloseAllFiles(model AS ModelFileInfo)
    DIM layer_idx AS INTEGER
    
    CloseFile model.token_embed_file
    CloseFile model.pos_embed_file
    CloseFile model.output_file
    CloseFile model.vocab_file
    CloseFile model.config_file
    
    FOR layer_idx = 0 TO model.num_layers - 1
        CloseFile model.layer_files(layer_idx)
    NEXT layer_idx
END SUB

' Read matrix dimensions from a file
SUB ReadMatrixDimensions(handle AS FileHandle, rows AS INTEGER, cols AS INTEGER)
    IF handle.is_open = 0 THEN
        IF OpenReadFile(handle) = 0 THEN
            rows = 0
            cols = 0
            RETURN
        END IF
    END IF
    
    ' Seek to the beginning of the file
    SEEK #handle.file_num, 1 ' 1-based indexing
    
    ' Read rows and columns (assuming they're stored as the first two integers)
    GET #handle.file_num, , rows
    GET #handle.file_num, , cols
END SUB

' Stream a portion of a matrix from disk
' Loads only a specified number of rows at a time to minimize memory usage
FUNCTION StreamMatrixRows(handle AS FileHandle, matrix AS Matrix, start_row AS INTEGER, num_rows AS INTEGER) AS INTEGER
    DIM rows AS INTEGER, cols AS INTEGER
    DIM bytes_per_value AS INTEGER
    DIM offset AS LONG
    DIM row AS INTEGER, col AS INTEGER
    DIM value AS INTEGER
    
    ' Make sure file is open
    IF handle.is_open = 0 THEN
        IF OpenReadFile(handle) = 0 THEN RETURN 0
    END IF
    
    ' Read matrix dimensions if not already known
    IF matrix.rows = 0 OR matrix.cols = 0 THEN
        ReadMatrixDimensions handle, rows, cols
        IF rows = 0 OR cols = 0 THEN RETURN 0 ' Error reading dimensions
        
        ' Initialize matrix if not already done
        IF matrix.rows = 0 OR matrix.cols = 0 OR matrix.rows <> rows OR matrix.cols <> cols THEN
            ' Free existing data if any
            IF matrix.rows > 0 AND matrix.cols > 0 THEN
                FreeMatrix matrix
            END IF
            InitMatrix matrix, rows, cols
        END IF
    ELSE
        rows = matrix.rows
        cols = matrix.cols
    END IF
    
    ' Verify bounds
    IF start_row < 0 OR start_row >= rows THEN RETURN 0
    IF start_row + num_rows > rows THEN
        num_rows = rows - start_row ' Adjust to read only available rows
    END IF
    
    ' Assume 1 byte per value (4-bit quantized packed as bytes)
    bytes_per_value = 1
    
    ' Calculate file offset (header + row offset)
    offset = HEADER_SIZE + (start_row * cols * bytes_per_value)
    
    ' Seek to the correct position
    SEEK #handle.file_num, offset + 1 ' Add 1 for 1-based file position
    
    ' Read the requested rows
    FOR row = 0 TO num_rows - 1
        FOR col = 0 TO cols - 1
            ' Read a byte from the file
            GET #handle.file_num, , value
            
            ' Store in the matrix
            matrix.data(start_row + row, col) = value
        NEXT col
    NEXT row
    
    FUNCTION = 1 ' Success
END FUNCTION

' Load an entire matrix from disk if it fits in memory
FUNCTION LoadMatrix(handle AS FileHandle, matrix AS Matrix) AS INTEGER
    DIM rows AS INTEGER, cols AS INTEGER
    
    ' Read dimensions
    ReadMatrixDimensions handle, rows, cols
    IF rows = 0 OR cols = 0 THEN RETURN 0 ' Error reading dimensions
    
    ' Initialize matrix
    InitMatrix matrix, rows, cols
    
    ' Stream all rows at once
    FUNCTION = StreamMatrixRows(handle, matrix, 0, rows)
END FUNCTION

' Stream a layer's weights from disk
SUB StreamLayerWeights(model AS ModelFileInfo, layer_idx AS INTEGER, Wq AS Matrix, Wk AS Matrix, Wv AS Matrix, Wo AS Matrix, _
                      W1 AS Matrix, W2 AS Matrix, W3 AS Matrix, _
                      LayerNorm1_gamma AS Matrix, LayerNorm1_beta AS Matrix, _
                      LayerNorm2_gamma AS Matrix, LayerNorm2_beta AS Matrix)
    DIM handle AS FileHandle
    DIM offset AS LONG
    DIM matrix_idx AS INTEGER
    DIM success AS INTEGER
    
    ' Use the appropriate layer file
    handle = model.layer_files(layer_idx)
    
    ' Make sure file is open
    IF handle.is_open = 0 THEN
        IF OpenReadFile(handle) = 0 THEN
            PRINT "Error: Could not open layer file for layer "; layer_idx
            RETURN
        END IF
    END IF
    
    ' Each layer file contains multiple matrices in sequence:
    ' 1. Wq - Query weights
    ' 2. Wk - Key weights
    ' 3. Wv - Value weights
    ' 4. Wo - Output projection weights
    ' 5. W1 - FFN first linear layer weights
    ' 6. W2 - FFN second linear layer weights
    ' 7. W3 - FFN gate weights
    ' 8. LayerNorm1_gamma
    ' 9. LayerNorm1_beta
    ' 10. LayerNorm2_gamma
    ' 11. LayerNorm2_beta
    
    ' Seek to the beginning of the file
    SEEK #handle.file_num, 1 ' 1-based indexing
    
    ' There's a header at the start indicating how many matrices and their dimensions
    DIM num_matrices AS INTEGER
    GET #handle.file_num, , num_matrices
    
    ' Now load each matrix in sequence
    ' We'll use a simple approach of reading one matrix after another
    ' This could be optimized further to read only what's needed
    
    ' 1. Load Wq
    success = LoadMatrix(handle, Wq)
    IF success = 0 THEN
        PRINT "Error loading Wq for layer "; layer_idx
        RETURN
    END IF
    
    ' 2. Load Wk
    success = LoadMatrix(handle, Wk)
    IF success = 0 THEN
        PRINT "Error loading Wk for layer "; layer_idx
        RETURN
    END IF
    
    ' 3. Load Wv
    success = LoadMatrix(handle, Wv)
    IF success = 0 THEN
        PRINT "Error loading Wv for layer "; layer_idx
        RETURN
    END IF
    
    ' 4. Load Wo
    success = LoadMatrix(handle, Wo)
    IF success = 0 THEN
        PRINT "Error loading Wo for layer "; layer_idx
        RETURN
    END IF
    
    ' 5. Load W1
    success = LoadMatrix(handle, W1)
    IF success = 0 THEN
        PRINT "Error loading W1 for layer "; layer_idx
        RETURN
    END IF
    
    ' 6. Load W2
    success = LoadMatrix(handle, W2)
    IF success = 0 THEN
        PRINT "Error loading W2 for layer "; layer_idx
        RETURN
    END IF
    
    ' 7. Load W3
    success = LoadMatrix(handle, W3)
    IF success = 0 THEN
        PRINT "Error loading W3 for layer "; layer_idx
        RETURN
    END IF
    
    ' 8. Load LayerNorm1_gamma
    success = LoadMatrix(handle, LayerNorm1_gamma)
    IF success = 0 THEN
        PRINT "Error loading LayerNorm1_gamma for layer "; layer_idx
        RETURN
    END IF
    
    ' 9. Load LayerNorm1_beta
    success = LoadMatrix(handle, LayerNorm1_beta)
    IF success = 0 THEN
        PRINT "Error loading LayerNorm1_beta for layer "; layer_idx
        RETURN
    END IF
    
    ' 10. Load LayerNorm2_gamma
    success = LoadMatrix(handle, LayerNorm2_gamma)
    IF success = 0 THEN
        PRINT "Error loading LayerNorm2_gamma for layer "; layer_idx
        RETURN
    END IF
    
    ' 11. Load LayerNorm2_beta
    success = LoadMatrix(handle, LayerNorm2_beta)
    IF success = 0 THEN
        PRINT "Error loading LayerNorm2_beta for layer "; layer_idx
        RETURN
    END IF
END SUB

' Load vocabulary from file
FUNCTION LoadModelVocabulary(model AS ModelFileInfo, vocab() AS STRING) AS INTEGER
    DIM handle AS FileHandle
    DIM token_idx AS INTEGER
    DIM line_buffer AS STRING * 256
    DIM vocab_size AS INTEGER
    
    handle = model.vocab_file
    
    ' Make sure file is open
    IF handle.is_open = 0 THEN
        IF OpenReadFile(handle) = 0 THEN RETURN 0
    END IF
    
    ' First line should contain vocabulary size
    LINE INPUT #handle.file_num, line_buffer
    vocab_size = VAL(line_buffer)
    
    ' Resize vocabulary array if needed
    REDIM vocab(vocab_size - 1) AS STRING
    
    ' Read each token
    FOR token_idx = 0 TO vocab_size - 1
        IF EOF(handle.file_num) THEN
            PRINT "Warning: Unexpected end of vocabulary file"
            RETURN token_idx ' Return the number of tokens read
        END IF
        
        LINE INPUT #handle.file_num, line_buffer
        vocab(token_idx) = RTRIM$(line_buffer)
    NEXT token_idx
    
    FUNCTION = vocab_size
END FUNCTION

' Load model configuration from file
SUB LoadModelConfig(model AS ModelFileInfo, BYREF cfg_embedding_dim AS INTEGER, BYREF cfg_num_heads AS INTEGER, BYREF cfg_num_layers AS INTEGER, _
                    BYREF cfg_context_length AS INTEGER, BYREF cfg_vocab_size AS INTEGER)
    DIM handle AS FileHandle
    DIM cfg_filename AS STRING
    handle = model.config_file

    cfg_filename = RTRIM$(handle.filename)
    IF INSTR(UCASE$(cfg_filename), "GPT2CFG.TXT") > 0 OR INSTR(UCASE$(cfg_filename), "TINYCFG.TXT") > 0 THEN
        DIM text_file AS INTEGER
        DIM line_buffer AS STRING
        DIM eq_pos AS INTEGER
        DIM key_text AS STRING
        DIM value_text AS STRING

        cfg_embedding_dim = 0
        cfg_num_heads = 0
        cfg_num_layers = 0
        cfg_context_length = 0
        cfg_vocab_size = 0

        text_file = FREEFILE
        ON ERROR GOTO text_config_done
        OPEN cfg_filename FOR INPUT AS #text_file

        WHILE EOF(text_file) = 0
            LINE INPUT #text_file, line_buffer
            line_buffer = TRIM$(line_buffer)
            IF line_buffer <> "" AND LEFT$(line_buffer, 1) <> "#" THEN
                eq_pos = INSTR(line_buffer, "=")
                IF eq_pos > 0 THEN
                    key_text = LCASE$(TRIM$(LEFT$(line_buffer, eq_pos - 1)))
                    value_text = TRIM$(MID$(line_buffer, eq_pos + 1))

                    SELECT CASE key_text
                        CASE "n_embd", "embedding_dim"
                            cfg_embedding_dim = VAL(value_text)
                        CASE "n_head", "num_heads"
                            cfg_num_heads = VAL(value_text)
                        CASE "n_layer", "num_layers"
                            cfg_num_layers = VAL(value_text)
                        CASE "n_positions", "context_length"
                            cfg_context_length = VAL(value_text)
                        CASE "vocab_size"
                            cfg_vocab_size = VAL(value_text)
                    END SELECT
                END IF
            END IF
        WEND
        CLOSE #text_file

text_config_done:
        ON ERROR GOTO 0
        IF cfg_embedding_dim < 1 THEN cfg_embedding_dim = 48
        IF cfg_num_heads < 1 THEN cfg_num_heads = 4
        IF cfg_num_layers < 1 THEN cfg_num_layers = model.num_layers
        IF cfg_context_length < 1 THEN cfg_context_length = 192
        IF cfg_vocab_size < 1 THEN cfg_vocab_size = 258
        model.num_layers = cfg_num_layers
        RETURN
    END IF
    
    ' Make sure file is open
    IF handle.is_open = 0 THEN
        IF OpenReadFile(handle) = 0 THEN
            PRINT "Error: Could not open config file"
            RETURN
        END IF
    END IF
    
    ' Seek to the beginning of the file
    SEEK #handle.file_num, 1 ' 1-based indexing
    
    ' Read configuration parameters
    GET #handle.file_num, , cfg_embedding_dim
    GET #handle.file_num, , cfg_num_heads
    GET #handle.file_num, , cfg_num_layers
    GET #handle.file_num, , cfg_context_length
    GET #handle.file_num, , cfg_vocab_size
    
    ' Update the model's num_layers
    model.num_layers = cfg_num_layers
END SUB

' Write a model configuration file (for saving trained models)
SUB WriteModelConfig(base_path AS STRING, cfg_embedding_dim AS INTEGER, cfg_num_heads AS INTEGER, cfg_num_layers AS INTEGER, _
                     cfg_context_length AS INTEGER, cfg_vocab_size AS INTEGER)
    DIM file_num AS INTEGER
    file_num = FREEFILE
    
    ' Create the config file
    ON ERROR GOTO WriteError
    OPEN base_path + "\CONFIG.BIN" FOR BINARY AS #file_num
    ON ERROR GOTO 0
    
    ' Write configuration parameters
    PUT #file_num, , cfg_embedding_dim
    PUT #file_num, , cfg_num_heads
    PUT #file_num, , cfg_num_layers
    PUT #file_num, , cfg_context_length
    PUT #file_num, , cfg_vocab_size
    
    CLOSE #file_num
    RETURN
    
WriteError:
    PRINT "Error writing config file"
    CLOSE #file_num
END SUB

' Save a matrix to a binary file
SUB SaveMatrixToFile(matrix AS Matrix, filename AS STRING)
    DIM file_num AS INTEGER
    DIM r AS INTEGER, c AS INTEGER
    
    file_num = FREEFILE
    
    ' Create the file
    ON ERROR GOTO SaveError
    OPEN filename FOR BINARY AS #file_num
    ON ERROR GOTO 0
    
    ' Write dimensions
    PUT #file_num, , matrix.rows
    PUT #file_num, , matrix.cols
    
    ' Write data (byte by byte)
    FOR r = 0 TO matrix.rows - 1
        FOR c = 0 TO matrix.cols - 1
            PUT #file_num, , matrix.data(r, c)
        NEXT c
    NEXT r
    
    CLOSE #file_num
    RETURN
    
SaveError:
    PRINT "Error saving matrix to file: "; filename
    CLOSE #file_num
END SUB

' Creates a diagnostic model from scratch with randomized weights
' This is useful for file-format checks without touching the production checkpoint
SUB CreateDiagnosticModel(base_path AS STRING, cfg_embedding_dim AS INTEGER, cfg_num_heads AS INTEGER, cfg_num_layers AS INTEGER, _
                          cfg_context_length AS INTEGER, cfg_vocab_size AS INTEGER)
    DIM layer_idx AS INTEGER
    DIM cmd AS STRING
    
    ' Create the base directory if it doesn't exist
    cmd = "MD " + CHR$(34) + base_path + CHR$(34) ' Create directory command
    SHELL cmd                                      ' Execute command (will fail silently if directory exists)
    
    ' Write configuration
    WriteModelConfig base_path, cfg_embedding_dim, cfg_num_heads, cfg_num_layers, cfg_context_length, cfg_vocab_size
    
    ' Create diagnostic vocabulary file
    DIM vocab_file AS INTEGER
    vocab_file = FREEFILE
    OPEN base_path + "\VOCAB.TXT" FOR OUTPUT AS #vocab_file
    PRINT #vocab_file, cfg_vocab_size
    
    DIM i AS INTEGER
    FOR i = 0 TO cfg_vocab_size - 1
        PRINT #vocab_file, "token_" + LTRIM$(STR$(i))
    NEXT i
    CLOSE #vocab_file
    
    ' Create diagnostic weight matrices and save them
    ' Token embeddings
    DIM token_embed AS Matrix
    InitMatrix token_embed, cfg_vocab_size, cfg_embedding_dim
    ' Fill with random values
    FOR i = 0 TO (cfg_vocab_size * cfg_embedding_dim) - 1
        DIM row AS INTEGER = i \ cfg_embedding_dim
        DIM col AS INTEGER = i MOD cfg_embedding_dim
        ' Generate values between -1 and 1, then quantize
        DIM random_value AS SINGLE = (RND - 0.5) * 2
        token_embed.data(row, col) = QuantizeLog(random_value).packed_value
    NEXT i
    SaveMatrixToFile token_embed, base_path + "\TOKEMB.BIN"
    FreeMatrix token_embed
    
    ' Positional embeddings
    DIM pos_embed AS Matrix
    InitMatrix pos_embed, cfg_context_length, cfg_embedding_dim
    ' Fill with random values
    FOR i = 0 TO (cfg_context_length * cfg_embedding_dim) - 1
        DIM row AS INTEGER = i \ cfg_embedding_dim
        DIM col AS INTEGER = i MOD cfg_embedding_dim
        DIM random_value AS SINGLE = (RND - 0.5) * 2
        pos_embed.data(row, col) = QuantizeLog(random_value).packed_value
    NEXT i
    SaveMatrixToFile pos_embed, base_path + "\POSEMB.BIN"
    FreeMatrix pos_embed
    
    ' For each layer, create all required weight matrices
    FOR layer_idx = 0 TO cfg_num_layers - 1
        DIM layer_stem AS STRING = base_path + "\L" + LTRIM$(STR$(layer_idx))
        DIM layer_path AS STRING = layer_stem + ".BIN"
        DIM layer_file AS INTEGER = FREEFILE
        
        ' Create the layer file
        OPEN layer_path FOR BINARY AS #layer_file
        
        ' Write number of matrices in this file
        DIM num_matrices AS INTEGER = 11 ' Wq, Wk, Wv, Wo, W1, W2, W3, LN1_g, LN1_b, LN2_g, LN2_b
        PUT #layer_file, , num_matrices
        
        ' Close for now - we'll let SaveMatrixToFile handle the rest
        CLOSE #layer_file
        
        ' Create all the weight matrices for this layer with random values
        ' Attention weights
        DIM head_width AS INTEGER = cfg_embedding_dim \ cfg_num_heads
        
        ' Query weights
        DIM Wq AS Matrix
        InitMatrix Wq, cfg_embedding_dim, cfg_embedding_dim
        FOR i = 0 TO (cfg_embedding_dim * cfg_embedding_dim) - 1
            DIM row AS INTEGER = i \ cfg_embedding_dim
            DIM col AS INTEGER = i MOD cfg_embedding_dim
            DIM random_value AS SINGLE = (RND - 0.5) * 2
            Wq.data(row, col) = QuantizeLog(random_value).packed_value
        NEXT i
        SaveMatrixToFile Wq, layer_stem + ".WQ" ' Temporary file
        FreeMatrix Wq
        
        ' Key weights
        DIM Wk AS Matrix
        InitMatrix Wk, cfg_embedding_dim, cfg_embedding_dim
        FOR i = 0 TO (cfg_embedding_dim * cfg_embedding_dim) - 1
            DIM row AS INTEGER = i \ cfg_embedding_dim
            DIM col AS INTEGER = i MOD cfg_embedding_dim
            DIM random_value AS SINGLE = (RND - 0.5) * 2
            Wk.data(row, col) = QuantizeLog(random_value).packed_value
        NEXT i
        SaveMatrixToFile Wk, layer_stem + ".WK" ' Temporary file
        FreeMatrix Wk
        
        ' Value weights
        DIM Wv AS Matrix
        InitMatrix Wv, cfg_embedding_dim, cfg_embedding_dim
        FOR i = 0 TO (cfg_embedding_dim * cfg_embedding_dim) - 1
            DIM row AS INTEGER = i \ cfg_embedding_dim
            DIM col AS INTEGER = i MOD cfg_embedding_dim
            DIM random_value AS SINGLE = (RND - 0.5) * 2
            Wv.data(row, col) = QuantizeLog(random_value).packed_value
        NEXT i
        SaveMatrixToFile Wv, layer_stem + ".WV" ' Temporary file
        FreeMatrix Wv
        
        ' Output projection weights
        DIM Wo AS Matrix
        InitMatrix Wo, cfg_embedding_dim, cfg_embedding_dim
        FOR i = 0 TO (cfg_embedding_dim * cfg_embedding_dim) - 1
            DIM row AS INTEGER = i \ cfg_embedding_dim
            DIM col AS INTEGER = i MOD cfg_embedding_dim
            DIM random_value AS SINGLE = (RND - 0.5) * 2
            Wo.data(row, col) = QuantizeLog(random_value).packed_value
        NEXT i
        SaveMatrixToFile Wo, layer_stem + ".WO" ' Temporary file
        FreeMatrix Wo
        
        ' FFN weights
        DIM intermediate_dim AS INTEGER = cfg_embedding_dim * 4
        
        ' W1
        DIM W1 AS Matrix
        InitMatrix W1, cfg_embedding_dim, intermediate_dim
        FOR i = 0 TO (cfg_embedding_dim * intermediate_dim) - 1
            DIM row AS INTEGER = i \ intermediate_dim
            DIM col AS INTEGER = i MOD intermediate_dim
            DIM random_value AS SINGLE = (RND - 0.5) * 2
            W1.data(row, col) = QuantizeLog(random_value).packed_value
        NEXT i
        SaveMatrixToFile W1, layer_stem + ".W1" ' Temporary file
        FreeMatrix W1
        
        ' W2
        DIM W2 AS Matrix
        InitMatrix W2, intermediate_dim, cfg_embedding_dim
        FOR i = 0 TO (intermediate_dim * cfg_embedding_dim) - 1
            DIM row AS INTEGER = i \ cfg_embedding_dim
            DIM col AS INTEGER = i MOD cfg_embedding_dim
            DIM random_value AS SINGLE = (RND - 0.5) * 2
            W2.data(row, col) = QuantizeLog(random_value).packed_value
        NEXT i
        SaveMatrixToFile W2, layer_stem + ".W2" ' Temporary file
        FreeMatrix W2
        
        ' W3 (gate)
        DIM W3 AS Matrix
        InitMatrix W3, cfg_embedding_dim, intermediate_dim
        FOR i = 0 TO (cfg_embedding_dim * intermediate_dim) - 1
            DIM row AS INTEGER = i \ intermediate_dim
            DIM col AS INTEGER = i MOD intermediate_dim
            DIM random_value AS SINGLE = (RND - 0.5) * 2
            W3.data(row, col) = QuantizeLog(random_value).packed_value
        NEXT i
        SaveMatrixToFile W3, layer_stem + ".W3" ' Temporary file
        FreeMatrix W3
        
        ' Layer norm parameters
        ' LN1 gamma
        DIM LN1_gamma AS Matrix
        InitMatrix LN1_gamma, cfg_embedding_dim, 1
        FOR i = 0 TO cfg_embedding_dim - 1
            LN1_gamma.data(i, 0) = QuantizeLog(1.0).packed_value ' Initialize with ones
        NEXT i
        SaveMatrixToFile LN1_gamma, layer_stem + ".G1" ' Temporary file
        FreeMatrix LN1_gamma
        
        ' LN1 beta
        DIM LN1_beta AS Matrix
        InitMatrix LN1_beta, cfg_embedding_dim, 1
        FOR i = 0 TO cfg_embedding_dim - 1
            LN1_beta.data(i, 0) = QuantizeLog(0.0).packed_value ' Initialize with zeros
        NEXT i
        SaveMatrixToFile LN1_beta, layer_stem + ".B1" ' Temporary file
        FreeMatrix LN1_beta
        
        ' LN2 gamma
        DIM LN2_gamma AS Matrix
        InitMatrix LN2_gamma, cfg_embedding_dim, 1
        FOR i = 0 TO cfg_embedding_dim - 1
            LN2_gamma.data(i, 0) = QuantizeLog(1.0).packed_value ' Initialize with ones
        NEXT i
        SaveMatrixToFile LN2_gamma, layer_stem + ".G2" ' Temporary file
        FreeMatrix LN2_gamma
        
        ' LN2 beta
        DIM LN2_beta AS Matrix
        InitMatrix LN2_beta, cfg_embedding_dim, 1
        FOR i = 0 TO cfg_embedding_dim - 1
            LN2_beta.data(i, 0) = QuantizeLog(0.0).packed_value ' Initialize with zeros
        NEXT i
        SaveMatrixToFile LN2_beta, layer_stem + ".B2" ' Temporary file
        FreeMatrix LN2_beta
        
        ' Now combine all these files into one layer file
        ' Use a system command to concatenate files
        ' This is OS-specific and may need adjustment
        cmd = "COPY /B " + layer_path + "+" + _
              layer_stem + ".WQ+" + _
              layer_stem + ".WK+" + _
              layer_stem + ".WV+" + _
              layer_stem + ".WO+" + _
              layer_stem + ".W1+" + _
              layer_stem + ".W2+" + _
              layer_stem + ".W3+" + _
              layer_stem + ".G1+" + _
              layer_stem + ".B1+" + _
              layer_stem + ".G2+" + _
              layer_stem + ".B2 " + _
              layer_stem + ".TMP"
        SHELL cmd
        
        ' Rename the tmp file to the final layer file
        cmd = "MOVE /Y " + layer_stem + ".TMP " + layer_path
        SHELL cmd
        
        ' Delete temporary files
        cmd = "DEL " + layer_stem + ".WQ " + _
              layer_stem + ".WK " + _
              layer_stem + ".WV " + _
              layer_stem + ".WO " + _
              layer_stem + ".W1 " + _
              layer_stem + ".W2 " + _
              layer_stem + ".W3 " + _
              layer_stem + ".G1 " + _
              layer_stem + ".B1 " + _
              layer_stem + ".G2 " + _
              layer_stem + ".B2"
        SHELL cmd
    NEXT layer_idx
    
    ' Create output layer weights
    DIM output_layer AS Matrix
    InitMatrix output_layer, cfg_embedding_dim, cfg_vocab_size
    ' Fill with random values
    FOR i = 0 TO (cfg_embedding_dim * cfg_vocab_size) - 1
        DIM row AS INTEGER = i \ cfg_vocab_size
        DIM col AS INTEGER = i MOD cfg_vocab_size
        DIM random_value AS SINGLE = (RND - 0.5) * 2
        output_layer.data(row, col) = QuantizeLog(random_value).packed_value
    NEXT i
    SaveMatrixToFile output_layer, base_path + "\OUTPUT.BIN"
    FreeMatrix output_layer
    
    PRINT "Created diagnostic model in directory: "; base_path
    PRINT "Embedding dimension: "; cfg_embedding_dim
    PRINT "Number of heads: "; cfg_num_heads
    PRINT "Number of layers: "; cfg_num_layers
    PRINT "Context length: "; cfg_context_length
    PRINT "Vocabulary size: "; cfg_vocab_size
END SUB
