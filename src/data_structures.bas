' *******************************************************
' * Data Structures for GPT-2 BASIC                     *
' *******************************************************
' * This module defines the core data structures used   *
' * throughout the implementation, including matrices,   *
' * tensors, and model configuration.                   *
' *                                                     *
' * It provides initialization, memory management, and  *
' * basic operations for these structures.              *
' *******************************************************

' *******************************************************
' * Matrix Structure and Basic Operations               *
' *******************************************************

' Basic matrix structure
TYPE Matrix
    rows AS INTEGER         ' Number of rows
    cols AS INTEGER         ' Number of columns
    data() AS SINGLE        ' Matrix data (row-major order)
END TYPE

' Initialize a matrix with specified dimensions
SUB InitMatrix(BYREF mat AS Matrix, rows AS INTEGER, cols AS INTEGER)
    ' Set dimensions
    mat.rows = rows
    mat.cols = cols
    
    ' Allocate memory for matrix data
    REDIM mat.data(0 TO rows - 1, 0 TO cols - 1)
END SUB

' Free the memory used by a matrix
SUB FreeMatrix(BYREF mat AS Matrix)
    ' Clear the matrix data array
    ERASE mat.data
    
    ' Reset dimensions
    mat.rows = 0
    mat.cols = 0
END SUB

' Zero a matrix (set all elements to 0)
SUB ZeroMatrix(BYREF mat AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    
    FOR i = 0 TO mat.rows - 1
        FOR j = 0 TO mat.cols - 1
            mat.data(i, j) = 0.0
        NEXT j
    NEXT i
END SUB

' Copy the contents of one matrix to another
SUB CopyMatrix(src AS Matrix, BYREF dst AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    
    ' Initialize destination matrix with same dimensions
    InitMatrix(dst, src.rows, src.cols)
    
    ' Copy data
    FOR i = 0 TO src.rows - 1
        FOR j = 0 TO src.cols - 1
            dst.data(i, j) = src.data(i, j)
        NEXT j
    NEXT i
END SUB

' Multiply a matrix by a scalar
SUB ScaleMatrix(BYREF mat AS Matrix, scale_factor AS SINGLE)
    DIM i AS INTEGER, j AS INTEGER
    
    FOR i = 0 TO mat.rows - 1
        FOR j = 0 TO mat.cols - 1
            mat.data(i, j) = mat.data(i, j) * scale_factor
        NEXT j
    NEXT i
END SUB

' Print a matrix (for debugging)
SUB PrintMatrix(mat AS Matrix, name AS STRING)
    DIM i AS INTEGER, j AS INTEGER
    
    PRINT "Matrix "; name; " ("; mat.rows; "x"; mat.cols; "):"
    
    ' For large matrices, only print a subset
    DIM max_rows AS INTEGER, max_cols AS INTEGER
    max_rows = MIN(mat.rows, 6)
    max_cols = MIN(mat.cols, 6)
    
    FOR i = 0 TO max_rows - 1
        FOR j = 0 TO max_cols - 1
            PRINT FORMAT(mat.data(i, j), "0.000"); " ";
        NEXT j
        
        ' If we truncated columns, indicate it
        IF max_cols < mat.cols THEN
            PRINT "...";
        END IF
        
        PRINT
    NEXT i
    
    ' If we truncated rows, indicate it
    IF max_rows < mat.rows THEN
        PRINT "..."
    END IF
    
    PRINT
END SUB

' *******************************************************
' * Vector Operations                                   *
' *******************************************************

' Note: Vectors are represented as Nx1 or 1xN matrices

' Compute dot product of two vectors
FUNCTION VectorDotProduct(a AS Matrix, b AS Matrix) AS SINGLE
    DIM sum AS SINGLE, i AS INTEGER
    
    ' Check that dimensions are compatible
    IF a.rows * a.cols <> b.rows * b.cols THEN
        PRINT "ERROR: Vector dimensions don't match for dot product"
        RETURN 0.0
    END IF
    
    sum = 0.0
    
    ' Handle both row and column vectors
    IF a.rows = 1 THEN
        ' Row vectors
        FOR i = 0 TO a.cols - 1
            sum = sum + a.data(0, i) * b.data(0, i)
        NEXT i
    ELSE
        ' Column vectors
        FOR i = 0 TO a.rows - 1
            sum = sum + a.data(i, 0) * b.data(i, 0)
        NEXT i
    END IF
    
    RETURN sum
END FUNCTION

' Compute L2 norm (Euclidean length) of a vector
FUNCTION VectorNorm(vec AS Matrix) AS SINGLE
    DIM sum_sq AS SINGLE, i AS INTEGER
    
    sum_sq = 0.0
    
    ' Handle both row and column vectors
    IF vec.rows = 1 THEN
        ' Row vector
        FOR i = 0 TO vec.cols - 1
            sum_sq = sum_sq + vec.data(0, i) * vec.data(0, i)
        NEXT i
    ELSE
        ' Column vector
        FOR i = 0 TO vec.rows - 1
            sum_sq = sum_sq + vec.data(i, 0) * vec.data(i, 0)
        NEXT i
    END IF
    
    RETURN SQR(sum_sq)
END FUNCTION

' *******************************************************
' * Tensor Structure and Operations                     *
' *******************************************************

' Tensor structure for higher-dimensional data
TYPE Tensor
    dims(0 TO 3) AS INTEGER   ' Dimensions (up to 4D tensor)
    n_dims AS INTEGER         ' Number of dimensions (1-4)
    size AS INTEGER           ' Total number of elements
    data() AS SINGLE          ' Flattened tensor data
END TYPE

' Initialize a tensor with specified dimensions
SUB InitTensor(BYREF tensor AS Tensor, dim1 AS INTEGER, dim2 AS INTEGER, dim3 AS INTEGER, dim4 AS INTEGER)
    ' Set dimensions based on non-zero values
    IF dim4 > 0 THEN
        tensor.n_dims = 4
        tensor.dims(0) = dim1
        tensor.dims(1) = dim2
        tensor.dims(2) = dim3
        tensor.dims(3) = dim4
        tensor.size = dim1 * dim2 * dim3 * dim4
    ELSEIF dim3 > 0 THEN
        tensor.n_dims = 3
        tensor.dims(0) = dim1
        tensor.dims(1) = dim2
        tensor.dims(2) = dim3
        tensor.dims(3) = 0
        tensor.size = dim1 * dim2 * dim3
    ELSEIF dim2 > 0 THEN
        tensor.n_dims = 2
        tensor.dims(0) = dim1
        tensor.dims(1) = dim2
        tensor.dims(2) = 0
        tensor.dims(3) = 0
        tensor.size = dim1 * dim2
    ELSE
        tensor.n_dims = 1
        tensor.dims(0) = dim1
        tensor.dims(1) = 0
        tensor.dims(2) = 0
        tensor.dims(3) = 0
        tensor.size = dim1
    END IF
    
    ' Allocate memory for tensor data
    REDIM tensor.data(0 TO tensor.size - 1)
END SUB

' Free the memory used by a tensor
SUB FreeTensor(BYREF tensor AS Tensor)
    ' Clear the tensor data array
    ERASE tensor.data
    
    ' Reset dimensions
    tensor.n_dims = 0
    tensor.size = 0
    tensor.dims(0) = 0
    tensor.dims(1) = 0
    tensor.dims(2) = 0
    tensor.dims(3) = 0
END SUB

' Zero a tensor (set all elements to 0)
SUB ZeroTensor(BYREF tensor AS Tensor)
    DIM i AS INTEGER
    
    FOR i = 0 TO tensor.size - 1
        tensor.data(i) = 0.0
    NEXT i
END SUB

' Calculate linear index from multi-dimensional indices
FUNCTION TensorIndex(tensor AS Tensor, idx0 AS INTEGER, idx1 AS INTEGER, idx2 AS INTEGER, idx3 AS INTEGER) AS INTEGER
    DIM linear_idx AS INTEGER
    
    SELECT CASE tensor.n_dims
        CASE 1:
            linear_idx = idx0
        CASE 2:
            linear_idx = idx0 * tensor.dims(1) + idx1
        CASE 3:
            linear_idx = (idx0 * tensor.dims(1) + idx1) * tensor.dims(2) + idx2
        CASE 4:
            linear_idx = ((idx0 * tensor.dims(1) + idx1) * tensor.dims(2) + idx2) * tensor.dims(3) + idx3
    END SELECT
    
    RETURN linear_idx
END FUNCTION

' Get value at specified indices
FUNCTION GetTensorValue(tensor AS Tensor, idx0 AS INTEGER, idx1 AS INTEGER, idx2 AS INTEGER, idx3 AS INTEGER) AS SINGLE
    DIM idx AS INTEGER
    
    idx = TensorIndex(tensor, idx0, idx1, idx2, idx3)
    RETURN tensor.data(idx)
END FUNCTION

' Set value at specified indices
SUB SetTensorValue(BYREF tensor AS Tensor, idx0 AS INTEGER, idx1 AS INTEGER, idx2 AS INTEGER, idx3 AS INTEGER, value AS SINGLE)
    DIM idx AS INTEGER
    
    idx = TensorIndex(tensor, idx0, idx1, idx2, idx3)
    tensor.data(idx) = value
END SUB

' Extract a matrix (2D slice) from a tensor
SUB ExtractMatrixFromTensor(tensor AS Tensor, BYREF mat AS Matrix, dim1_idx AS INTEGER, dim2_idx AS INTEGER)
    DIM i AS INTEGER, j AS INTEGER, tensor_idx AS INTEGER
    
    ' Determine which dimensions to extract based on tensor dimensionality
    SELECT CASE tensor.n_dims
        CASE 2:
            ' The tensor is already a matrix
            CopyMatrix(tensor, mat)
            
        CASE 3:
            ' Extract a matrix by fixing one dimension
            IF dim1_idx >= 0 THEN
                ' Fix first dimension, extract dims 1 & 2
                InitMatrix(mat, tensor.dims(1), tensor.dims(2))
                FOR i = 0 TO tensor.dims(1) - 1
                    FOR j = 0 TO tensor.dims(2) - 1
                        tensor_idx = TensorIndex(tensor, dim1_idx, i, j, 0)
                        mat.data(i, j) = tensor.data(tensor_idx)
                    NEXT j
                NEXT i
            ELSE
                ' Fix last dimension, extract dims 0 & 1
                InitMatrix(mat, tensor.dims(0), tensor.dims(1))
                FOR i = 0 TO tensor.dims(0) - 1
                    FOR j = 0 TO tensor.dims(1) - 1
                        tensor_idx = TensorIndex(tensor, i, j, dim2_idx, 0)
                        mat.data(i, j) = tensor.data(tensor_idx)
                    NEXT j
                NEXT i
            END IF
            
        CASE 4:
            ' Extract a matrix by fixing two dimensions
            InitMatrix(mat, tensor.dims(2), tensor.dims(3))
            FOR i = 0 TO tensor.dims(2) - 1
                FOR j = 0 TO tensor.dims(3) - 1
                    tensor_idx = TensorIndex(tensor, dim1_idx, dim2_idx, i, j)
                    mat.data(i, j) = tensor.data(tensor_idx)
                NEXT j
            NEXT i
    END SELECT
END SUB

' *******************************************************
' * Model Configuration                                 *
' *******************************************************

' GPT-2 model configuration
TYPE ModelConfig
    vocab_size AS INTEGER         ' Size of vocabulary
    n_positions AS INTEGER        ' Maximum sequence length
    n_ctx AS INTEGER              ' Context window size
    n_embd AS INTEGER             ' Embedding dimension
    n_layer AS INTEGER            ' Number of transformer layers
    n_head AS INTEGER             ' Number of attention heads
    precision AS INTEGER          ' Precision level (4, 8, 16, 32 bits)
    use_block_sparse AS INTEGER   ' Whether to use block-sparse attention
    block_size AS INTEGER         ' Block size for sparse attention
    use_fp16 AS INTEGER           ' Whether to use 16-bit floating point
    attention_window AS INTEGER   ' Local attention window size
END TYPE

' Initialize model configuration with default values
SUB InitModelConfig(BYREF config AS ModelConfig)
    ' Small GPT-2 defaults with 486-era optimizations
    config.vocab_size = 16384     ' Reduced from 50257
    config.n_positions = 512      ' Reduced from 1024
    config.n_ctx = 512            ' Reduced from 1024
    config.n_embd = 256           ' Reduced from 768
    config.n_layer = 6            ' Reduced from 12
    config.n_head = 8             ' Reduced from 12
    config.precision = 8          ' 8-bit precision by default
    config.use_block_sparse = 1   ' Enable block-sparse attention
    config.block_size = 32        ' 32x32 blocks
    config.use_fp16 = 0           ' No 16-bit support on 486
    config.attention_window = 64  ' Local attention window
END SUB

' Print model configuration
SUB PrintModelConfig(config AS ModelConfig)
    PRINT "GPT-2 Model Configuration:"
    PRINT "-------------------------"
    PRINT "Vocabulary size  : "; config.vocab_size
    PRINT "Max positions    : "; config.n_positions
    PRINT "Context window   : "; config.n_ctx
    PRINT "Embedding dim    : "; config.n_embd
    PRINT "Layers           : "; config.n_layer
    PRINT "Attention heads  : "; config.n_head
    PRINT "Precision        : "; config.precision; "-bit"
    PRINT "Block-sparse     : "; IIF(config.use_block_sparse, "Enabled", "Disabled")
    IF config.use_block_sparse THEN
        PRINT "Block size       : "; config.block_size; "x"; config.block_size
    END IF
    PRINT "Memory required  : ~"; EstimateModelMemory(config) / (1024 * 1024); " MB"
    PRINT
END SUB

' Estimate memory required for model
FUNCTION EstimateModelMemory(config AS ModelConfig) AS LONG
    DIM memory AS LONG
    DIM bytes_per_param AS INTEGER
    
    ' Determine bytes per parameter based on precision
    SELECT CASE config.precision
        CASE 4:  bytes_per_param = 1 ' 8 values per byte
        CASE 8:  bytes_per_param = 1
        CASE 16: bytes_per_param = 2
        CASE 32: bytes_per_param = 4
        CASE ELSE: bytes_per_param = 4
    END SELECT
    
    ' Parameter counts:
    ' - Token embeddings: vocab_size * n_embd
    ' - Position embeddings: n_positions * n_embd
    ' - Each layer:
    '   - Attention: 3 * n_embd * n_embd (query, key, value projections)
    '   - Attention output: n_embd * n_embd
    '   - FFN: 4 * n_embd * n_embd + n_embd * n_embd
    ' - Output projection: n_embd * vocab_size
    
    memory = config.vocab_size * config.n_embd ' Token embeddings
    memory = memory + config.n_positions * config.n_embd ' Position embeddings
    
    ' Layer parameters
    DIM layer_params AS LONG
    layer_params = 3 * config.n_embd * config.n_embd ' QKV projections
    layer_params = layer_params + config.n_embd * config.n_embd ' Attention output
    layer_params = layer_params + 4 * config.n_embd * config.n_embd ' FFN first layer
    layer_params = layer_params + config.n_embd * config.n_embd ' FFN second layer
    memory = memory + layer_params * config.n_layer
    
    ' Output projection
    memory = memory + config.n_embd * config.vocab_size
    
    ' Convert to bytes
    memory = memory * bytes_per_param
    
    ' Account for sparse attention if enabled (rough estimate)
    IF config.use_block_sparse THEN
        ' Sparse attention typically uses ~25-40% of dense memory
        ' Only affects the attention matrices (QK^T) during computation
        ' We don't count this in parameter storage, but in runtime memory
    END IF
    
    ' Add overhead for data structures, code, etc.
    memory = memory + 2 * 1024 * 1024 ' 2MB overhead
    
    RETURN memory
END FUNCTION

' Helper function for conditional expressions
FUNCTION IIF(condition AS INTEGER, true_val AS STRING, false_val AS STRING) AS STRING
    IF condition THEN
        RETURN true_val
    ELSE
        RETURN false_val
    END IF
END FUNCTION

' *******************************************************
' * Testing functions for data structures              *
' *******************************************************

' Test basic matrix operations
SUB TestMatrixOperations()
    DIM mat1 AS Matrix, mat2 AS Matrix, mat3 AS Matrix
    DIM i AS INTEGER, j AS INTEGER
    
    PRINT "Testing matrix operations..."
    
    ' Initialize matrices
    InitMatrix(mat1, 3, 4)
    InitMatrix(mat2, 3, 4)
    
    ' Fill with test data
    FOR i = 0 TO 2
        FOR j = 0 TO 3
            mat1.data(i, j) = i + j
            mat2.data(i, j) = i * j
        NEXT j
    NEXT i
    
    ' Test matrix operations
    PrintMatrix(mat1, "mat1")
    PrintMatrix(mat2, "mat2")
    
    ' Test copy
    CopyMatrix(mat1, mat3)
    PrintMatrix(mat3, "mat3 (copy of mat1)")
    
    ' Test scaling
    ScaleMatrix(mat3, 2.0)
    PrintMatrix(mat3, "mat3 after scaling by 2.0")
    
    ' Clean up
    FreeMatrix(mat1)
    FreeMatrix(mat2)
    FreeMatrix(mat3)
END SUB

' Test tensor operations
SUB TestTensorOperations()
    DIM tensor AS Tensor
    DIM mat AS Matrix
    DIM i AS INTEGER, j AS INTEGER, k AS INTEGER
    
    PRINT "Testing tensor operations..."
    
    ' Initialize a 3D tensor (2x3x4)
    InitTensor(tensor, 2, 3, 4, 0)
    
    ' Fill with test data
    FOR i = 0 TO 1
        FOR j = 0 TO 2
            FOR k = 0 TO 3
                SetTensorValue(tensor, i, j, k, 0, i * 100 + j * 10 + k)
            NEXT k
        NEXT j
    NEXT i
    
    ' Print some values
    PRINT "Tensor dimensions: "; tensor.dims(0); "x"; tensor.dims(1); "x"; tensor.dims(2)
    PRINT "Some tensor values:"
    PRINT "tensor(0,0,0) = "; GetTensorValue(tensor, 0, 0, 0, 0)
    PRINT "tensor(1,2,3) = "; GetTensorValue(tensor, 1, 2, 3, 0)
    
    ' Extract a matrix slice
    ExtractMatrixFromTensor(tensor, mat, 1, -1)
    PrintMatrix(mat, "Matrix slice from tensor (fixing first dimension=1)")
    
    ' Clean up
    FreeTensor(tensor)
    FreeMatrix(mat)
END SUB

' Test model configuration
SUB TestModelConfig()
    DIM config AS ModelConfig
    
    PRINT "Testing model configuration..."
    
    ' Initialize with defaults
    InitModelConfig(config)
    
    ' Print configuration
    PrintModelConfig(config)
    
    ' Try different model sizes
    PRINT "Memory estimates for different model sizes:"
    
    config.n_embd = 128
    config.n_layer = 4
    PRINT "Tiny model (128d, 4L): "; EstimateModelMemory(config) / (1024 * 1024); " MB"
    
    config.n_embd = 256
    config.n_layer = 6
    PRINT "Small model (256d, 6L): "; EstimateModelMemory(config) / (1024 * 1024); " MB"
    
    config.n_embd = 512
    config.n_layer = 8
    PRINT "Medium model (512d, 8L): "; EstimateModelMemory(config) / (1024 * 1024); " MB"
    
    config.n_embd = 768
    config.n_layer = 12
    PRINT "Standard GPT-2 (768d, 12L): "; EstimateModelMemory(config) / (1024 * 1024); " MB"
END SUB

' Main test routine
SUB TestDataStructures()
    PRINT "Testing Data Structures Module"
    PRINT "=============================="
    PRINT
    
    TestMatrixOperations()
    PRINT
    TestTensorOperations()
    PRINT
    TestModelConfig()
END SUB
