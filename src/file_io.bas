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

' Initialize the model file information
SUB InitModelFiles(model AS ModelFileInfo, base_path AS STRING)
    DIM layer_idx AS INTEGER
    
    ' Initialize file handles with proper paths
    InitFileHandle model.token_embed_file, base_path + "/token_embed.bin"
    InitFileHandle model.pos_embed_file, base_path + "/pos_embed.bin"
    InitFileHandle model.output_file, base_path + "/output_layer.bin"
    InitFileHandle model.vocab_file, base_path + "/vocabulary.txt"
    InitFileHandle model.config_file, base_path + "/config.bin"
    
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
        InitFileHandle model.layer_files(layer_idx), base_path + "/layer_" + LTRIM$(STR$(layer_idx)) + ".bin"
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
    IF NOT handle.is_open THEN
        IF NOT OpenReadFile(handle) THEN
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
    IF NOT handle.is_open THEN
        IF NOT OpenReadFile(handle) THEN RETURN 0
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
    IF NOT handle.is_open THEN
        IF NOT OpenReadFile(handle) THEN 
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
    IF NOT success THEN
        PRINT "Error loading Wq for layer "; layer_idx
        RETURN
    END IF
    
    ' 2. Load Wk
    success = LoadMatrix(handle, Wk)
    IF NOT success THEN
        PRINT "Error loading Wk for layer "; layer_idx
        RETURN
    END IF
    
    ' 3. Load Wv
    success = LoadMatrix(handle, Wv)
    IF NOT success THEN
        PRINT "Error loading Wv for layer "; layer_idx
        RETURN
    END IF
    
    ' 4. Load Wo
    success = LoadMatrix(handle, Wo)
    IF NOT success THEN
        PRINT "Error loading Wo for layer "; layer_idx
        RETURN
    END IF
    
    ' 5. Load W1
    success = LoadMatrix(handle, W1)
    IF NOT success THEN
        PRINT "Error loading W1 for layer "; layer_idx
        RETURN
    END IF
    
    ' 6. Load W2
    success = LoadMatrix(handle, W2)
    IF NOT success THEN
        PRINT "Error loading W2 for layer "; layer_idx
        RETURN
    END IF
    
    ' 7. Load W3
    success = LoadMatrix(handle, W3)
    IF NOT success THEN
        PRINT "Error loading W3 for layer "; layer_idx
        RETURN
    END IF
    
    ' 8. Load LayerNorm1_gamma
    success = LoadMatrix(handle, LayerNorm1_gamma)
    IF NOT success THEN
        PRINT "Error loading LayerNorm1_gamma for layer "; layer_idx
        RETURN
    END IF
    
    ' 9. Load LayerNorm1_beta
    success = LoadMatrix(handle, LayerNorm1_beta)
    IF NOT success THEN
        PRINT "Error loading LayerNorm1_beta for layer "; layer_idx
        RETURN
    END IF
    
    ' 10. Load LayerNorm2_gamma
    success = LoadMatrix(handle, LayerNorm2_gamma)
    IF NOT success THEN
        PRINT "Error loading LayerNorm2_gamma for layer "; layer_idx
        RETURN
    END IF
    
    ' 11. Load LayerNorm2_beta
    success = LoadMatrix(handle, LayerNorm2_beta)
    IF NOT success THEN
        PRINT "Error loading LayerNorm2_beta for layer "; layer_idx
        RETURN
    END IF
END SUB

' Load vocabulary from file
FUNCTION LoadVocabulary(model AS ModelFileInfo, vocab() AS STRING) AS INTEGER
    DIM handle AS FileHandle
    DIM token_idx AS INTEGER
    DIM line_buffer AS STRING * 256
    DIM vocab_size AS INTEGER
    
    handle = model.vocab_file
    
    ' Make sure file is open
    IF NOT handle.is_open THEN
        IF NOT OpenReadFile(handle) THEN RETURN 0
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
SUB LoadModelConfig(model AS ModelFileInfo, embedding_dim AS INTEGER, num_heads AS INTEGER, num_layers AS INTEGER, _
                    context_length AS INTEGER, vocab_size AS INTEGER)
    DIM handle AS FileHandle
    handle = model.config_file
    
    ' Make sure file is open
    IF NOT handle.is_open THEN
        IF NOT OpenReadFile(handle) THEN 
            PRINT "Error: Could not open config file"
            RETURN
        END IF
    END IF
    
    ' Seek to the beginning of the file
    SEEK #handle.file_num, 1 ' 1-based indexing
    
    ' Read configuration parameters
    GET #handle.file_num, , embedding_dim
    GET #handle.file_num, , num_heads
    GET #handle.file_num, , num_layers
    GET #handle.file_num, , context_length
    GET #handle.file_num, , vocab_size
    
    ' Update the model's num_layers
    model.num_layers = num_layers
END SUB

' Write a model configuration file (for saving trained models)
SUB WriteModelConfig(base_path AS STRING, embedding_dim AS INTEGER, num_heads AS INTEGER, num_layers AS INTEGER, _
                     context_length AS INTEGER, vocab_size AS INTEGER)
    DIM file_num AS INTEGER
    file_num = FREEFILE
    
    ' Create the config file
    ON ERROR GOTO WriteError
    OPEN base_path + "/config.bin" FOR BINARY AS #file_num
    ON ERROR GOTO 0
    
    ' Write configuration parameters
    PUT #file_num, , embedding_dim
    PUT #file_num, , num_heads
    PUT #file_num, , num_layers
    PUT #file_num, , context_length
    PUT #file_num, , vocab_size
    
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

' Creates a model from scratch with randomized weights
' This is useful for testing without an actual trained model
SUB CreateDummyModel(base_path AS STRING, embedding_dim AS INTEGER, num_heads AS INTEGER, num_layers AS INTEGER, _
                     context_length AS INTEGER, vocab_size AS INTEGER)
    DIM layer_idx AS INTEGER
    DIM cmd AS STRING
    
    ' Create the base directory if it doesn't exist
    cmd = "MD " + CHR$(34) + base_path + CHR$(34) ' Create directory command
    SHELL cmd                                      ' Execute command (will fail silently if directory exists)
    
    ' Write configuration
    WriteModelConfig base_path, embedding_dim, num_heads, num_layers, context_length, vocab_size
    
    ' Create dummy vocabulary file
    DIM vocab_file AS INTEGER
    vocab_file = FREEFILE
    OPEN base_path + "/vocabulary.txt" FOR OUTPUT AS #vocab_file
    PRINT #vocab_file, vocab_size
    
    DIM i AS INTEGER
    FOR i = 0 TO vocab_size - 1
        PRINT #vocab_file, "token_" + LTRIM$(STR$(i))
    NEXT i
    CLOSE #vocab_file
    
    ' Create dummy weight matrices and save them
    ' Token embeddings
    DIM token_embed AS Matrix
    InitMatrix token_embed, vocab_size, embedding_dim
    ' Fill with random values
    FOR i = 0 TO (vocab_size * embedding_dim) - 1
        DIM row AS INTEGER = i \ embedding_dim
        DIM col AS INTEGER = i MOD embedding_dim
        ' Generate values between -1 and 1, then quantize
        DIM val AS SINGLE = (RND - 0.5) * 2
        token_embed.data(row, col) = QuantizeLog(val).packed_value
    NEXT i
    SaveMatrixToFile token_embed, base_path + "/token_embed.bin"
    FreeMatrix token_embed
    
    ' Positional embeddings
    DIM pos_embed AS Matrix
    InitMatrix pos_embed, context_length, embedding_dim
    ' Fill with random values
    FOR i = 0 TO (context_length * embedding_dim) - 1
        DIM row AS INTEGER = i \ embedding_dim
        DIM col AS INTEGER = i MOD embedding_dim
        DIM val AS SINGLE = (RND - 0.5) * 2
        pos_embed.data(row, col) = QuantizeLog(val).packed_value
    NEXT i
    SaveMatrixToFile pos_embed, base_path + "/pos_embed.bin"
    FreeMatrix pos_embed
    
    ' For each layer, create all required weight matrices
    FOR layer_idx = 0 TO num_layers - 1
        DIM layer_path AS STRING = base_path + "/layer_" + LTRIM$(STR$(layer_idx)) + ".bin"
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
        DIM head_dim AS INTEGER = embedding_dim \ num_heads
        
        ' Query weights
        DIM Wq AS Matrix
        InitMatrix Wq, embedding_dim, embedding_dim
        FOR i = 0 TO (embedding_dim * embedding_dim) - 1
            DIM row AS INTEGER = i \ embedding_dim
            DIM col AS INTEGER = i MOD embedding_dim
            DIM val AS SINGLE = (RND - 0.5) * 2
            Wq.data(row, col) = QuantizeLog(val).packed_value
        NEXT i
        SaveMatrixToFile Wq, layer_path + ".wq" ' Temporary file
        FreeMatrix Wq
        
        ' Key weights
        DIM Wk AS Matrix
        InitMatrix Wk, embedding_dim, embedding_dim
        FOR i = 0 TO (embedding_dim * embedding_dim) - 1
            DIM row AS INTEGER = i \ embedding_dim
            DIM col AS INTEGER = i MOD embedding_dim
            DIM val AS SINGLE = (RND - 0.5) * 2
            Wk.data(row, col) = QuantizeLog(val).packed_value
        NEXT i
        SaveMatrixToFile Wk, layer_path + ".wk" ' Temporary file
        FreeMatrix Wk
        
        ' Value weights
        DIM Wv AS Matrix
        InitMatrix Wv, embedding_dim, embedding_dim
        FOR i = 0 TO (embedding_dim * embedding_dim) - 1
            DIM row AS INTEGER = i \ embedding_dim
            DIM col AS INTEGER = i MOD embedding_dim
            DIM val AS SINGLE = (RND - 0.5) * 2
            Wv.data(row, col) = QuantizeLog(val).packed_value
        NEXT i
        SaveMatrixToFile Wv, layer_path + ".wv" ' Temporary file
        FreeMatrix Wv
        
        ' Output projection weights
        DIM Wo AS Matrix
        InitMatrix Wo, embedding_dim, embedding_dim
        FOR i = 0 TO (embedding_dim * embedding_dim) - 1
            DIM row AS INTEGER = i \ embedding_dim
            DIM col AS INTEGER = i MOD embedding_dim
            DIM val AS SINGLE = (RND - 0.5) * 2
            Wo.data(row, col) = QuantizeLog(val).packed_value
        NEXT i
        SaveMatrixToFile Wo, layer_path + ".wo" ' Temporary file
        FreeMatrix Wo
        
        ' FFN weights
        DIM intermediate_dim AS INTEGER = embedding_dim * 4
        
        ' W1
        DIM W1 AS Matrix
        InitMatrix W1, embedding_dim, intermediate_dim
        FOR i = 0 TO (embedding_dim * intermediate_dim) - 1
            DIM row AS INTEGER = i \ intermediate_dim
            DIM col AS INTEGER = i MOD intermediate_dim
            DIM val AS SINGLE = (RND - 0.5) * 2
            W1.data(row, col) = QuantizeLog(val).packed_value
        NEXT i
        SaveMatrixToFile W1, layer_path + ".w1" ' Temporary file
        FreeMatrix W1
        
        ' W2
        DIM W2 AS Matrix
        InitMatrix W2, intermediate_dim, embedding_dim
        FOR i = 0 TO (intermediate_dim * embedding_dim) - 1
            DIM row AS INTEGER = i \ embedding_dim
            DIM col AS INTEGER = i MOD embedding_dim
            DIM val AS SINGLE = (RND - 0.5) * 2
            W2.data(row, col) = QuantizeLog(val).packed_value
        NEXT i
        SaveMatrixToFile W2, layer_path + ".w2" ' Temporary file
        FreeMatrix W2
        
        ' W3 (gate)
        DIM W3 AS Matrix
        InitMatrix W3, embedding_dim, intermediate_dim
        FOR i = 0 TO (embedding_dim * intermediate_dim) - 1
            DIM row AS INTEGER = i \ intermediate_dim
            DIM col AS INTEGER = i MOD intermediate_dim
            DIM val AS SINGLE = (RND - 0.5) * 2
            W3.data(row, col) = QuantizeLog(val).packed_value
        NEXT i
        SaveMatrixToFile W3, layer_path + ".w3" ' Temporary file
        FreeMatrix W3
        
        ' Layer norm parameters
        ' LN1 gamma
        DIM LN1_gamma AS Matrix
        InitMatrix LN1_gamma, embedding_dim, 1
        FOR i = 0 TO embedding_dim - 1
            LN1_gamma.data(i, 0) = QuantizeLog(1.0).packed_value ' Initialize with ones
        NEXT i
        SaveMatrixToFile LN1_gamma, layer_path + ".ln1g" ' Temporary file
        FreeMatrix LN1_gamma
        
        ' LN1 beta
        DIM LN1_beta AS Matrix
        InitMatrix LN1_beta, embedding_dim, 1
        FOR i = 0 TO embedding_dim - 1
            LN1_beta.data(i, 0) = QuantizeLog(0.0).packed_value ' Initialize with zeros
        NEXT i
        SaveMatrixToFile LN1_beta, layer_path + ".ln1b" ' Temporary file
        FreeMatrix LN1_beta
        
        ' LN2 gamma
        DIM LN2_gamma AS Matrix
        InitMatrix LN2_gamma, embedding_dim, 1
        FOR i = 0 TO embedding_dim - 1
            LN2_gamma.data(i, 0) = QuantizeLog(1.0).packed_value ' Initialize with ones
        NEXT i
        SaveMatrixToFile LN2_gamma, layer_path + ".ln2g" ' Temporary file
        FreeMatrix LN2_gamma
        
        ' LN2 beta
        DIM LN2_beta AS Matrix
        InitMatrix LN2_beta, embedding_dim, 1
        FOR i = 0 TO embedding_dim - 1
            LN2_beta.data(i, 0) = QuantizeLog(0.0).packed_value ' Initialize with zeros
        NEXT i
        SaveMatrixToFile LN2_beta, layer_path + ".ln2b" ' Temporary file
        FreeMatrix LN2_beta
        
        ' Now combine all these files into one layer file
        ' Use a system command to concatenate files
        ' This is OS-specific and may need adjustment
        cmd = "COPY /B " + layer_path + "+" + _
              layer_path + ".wq+" + _
              layer_path + ".wk+" + _
              layer_path + ".wv+" + _
              layer_path + ".wo+" + _
              layer_path + ".w1+" + _
              layer_path + ".w2+" + _
              layer_path + ".w3+" + _
              layer_path + ".ln1g+" + _
              layer_path + ".ln1b+" + _
              layer_path + ".ln2g+" + _
              layer_path + ".ln2b " + _
              layer_path + ".tmp"
        SHELL cmd
        
        ' Rename the tmp file to the final layer file
        cmd = "MOVE /Y " + layer_path + ".tmp " + layer_path
        SHELL cmd
        
        ' Delete temporary files
        cmd = "DEL " + layer_path + ".wq " + _
              layer_path + ".wk " + _
              layer_path + ".wv " + _
              layer_path + ".wo " + _
              layer_path + ".w1 " + _
              layer_path + ".w2 " + _
              layer_path + ".w3 " + _
              layer_path + ".ln1g " + _
              layer_path + ".ln1b " + _
              layer_path + ".ln2g " + _
              layer_path + ".ln2b"
        SHELL cmd
    NEXT layer_idx
    
    ' Create output layer weights
    DIM output_layer AS Matrix
    InitMatrix output_layer, embedding_dim, vocab_size
    ' Fill with random values
    FOR i = 0 TO (embedding_dim * vocab_size) - 1
        DIM row AS INTEGER = i \ vocab_size
        DIM col AS INTEGER = i MOD vocab_size
        DIM val AS SINGLE = (RND - 0.5) * 2
        output_layer.data(row, col) = QuantizeLog(val).packed_value
    NEXT i
    SaveMatrixToFile output_layer, base_path + "/output_layer.bin"
    FreeMatrix output_layer
    
    PRINT "Created dummy model in directory: "; base_path
    PRINT "Embedding dimension: "; embedding_dim
    PRINT "Number of heads: "; num_heads
    PRINT "Number of layers: "; num_layers
    PRINT "Context length: "; context_length
    PRINT "Vocabulary size: "; vocab_size
END SUB
