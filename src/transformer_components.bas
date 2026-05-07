' BASIC implementation of transformer components for the GPT-2-like model.
' This file contains implementations for the attention mechanism,
' feed-forward networks, and layer normalization, using fixed-point arithmetic.

' Include necessary files
#INCLUDE "data_structures.bas"
#INCLUDE "quantization.bas" ' For LogQuantized type and dequantization to fixed-point
#INCLUDE "matrix_ops.bas" ' For matrix operations and fixed-point arithmetic, including fixed-point math functions
#INCLUDE "softmax_fixed.bas" ' For fixed-point softmax implementation
#INCLUDE "block_sparse.bas" ' For block-sparse attention optimization

' Define the dimensions for the transformer architecture
CONST EMBEDDING_DIM AS INTEGER = 128
CONST NUM_HEADS AS INTEGER = 4
CONST CONTEXT_LENGTH AS INTEGER = 128
CONST HEAD_DIM AS INTEGER = EMBEDDING_DIM \ NUM_HEADS

' Pre-calculate the attention scaling factor in fixed-point.
' Scale is 1 / sqrt(head_dim)
CONST ATTENTION_SCALE_FP AS INTEGER = 11585 ' 1 / sqrt(32) in 16.16 fixed point

' Define a very large negative fixed-point value for attention masking.
CONST NEG_INF_FP AS INTEGER = -32768

DECLARE FUNCTION ShouldUseBlockSparseAttention(context_rows AS INTEGER) AS INTEGER
DECLARE SUB ExtractHeadWeights(W_all AS Matrix, BYREF W_head AS Matrix, head_idx AS INTEGER, head_width AS INTEGER)
DECLARE SUB ConcatenateHeadOutput(BYREF concatenated AS Matrix, head_output AS Matrix, head_idx AS INTEGER, head_width AS INTEGER)
DECLARE SUB ApplyGELU(A AS Matrix, BYREF B AS Matrix)

' Decide when the sparse implementation is worth its setup overhead.
FUNCTION ShouldUseBlockSparseAttention(context_rows AS INTEGER) AS INTEGER
    IF context_rows >= 64 THEN
        RETURN 1
    END IF
    RETURN 0
END FUNCTION

' Function to implement the scaled dot-product attention mechanism.
' This is a simplified structure focusing on the flow.
' It needs to be adapted for block-sparse attention and optimized fixed-point math.
' Input: Query, Key, Value matrices (assumed to be LogQuantized data)
' Output: Output matrix (LogQuantized data)
SUB ScaledDotProductAttention (query_mat AS Matrix, key_mat AS Matrix, value_mat AS Matrix, BYREF output_mat AS Matrix)
    ' Assumes Query, Key, Value are appropriately shaped for a single attention head
    ' e.g., Query: (context_length, head_dim), Key: (context_length, head_dim), Value: (context_length, head_dim)
    ' where head_dim = embedding_dim / num_heads
    
    ' Determine whether to use dense or block-sparse attention based on context length
    ' For small contexts, dense attention may be more efficient
    ' For larger contexts, block-sparse attention saves memory and computation
    DIM use_sparse AS INTEGER = ShouldUseBlockSparseAttention(query_mat.rows)
    
    IF use_sparse = 1 THEN
        ' Use optimized block-sparse attention for larger contexts
        BlockSparseAttention query_mat, key_mat, value_mat, output_mat, 1
    ELSE
        ' For small contexts, use the original dense attention implementation
        ' Need a temporary matrix for scores: (context_length, context_length)
        DIM scores_mat AS Matrix
        InitMatrix scores_mat, query_mat.rows, key_mat.rows ' Query.rows = context_length, Key.rows = context_length
        
        ' Perform matrix multiplication: Scores = Query * Key^T
        MatrixMultiplyTransposeB query_mat, key_mat, scores_mat
    
        ' Scale the scores by dividing by sqrt(head_dim)
        DIM r AS INTEGER
        DIM c AS INTEGER
        FOR r = 0 TO scores_mat.rows - 1
            FOR c = 0 TO scores_mat.cols - 1
                ' Convert score to fixed-point
                DIM fp_score AS INTEGER = DequantizeLogToFixed(scores_mat.data(r, c)) ' Use direct dequantization to fixed-point
                
                ' Scale the score (fixed-point multiplication)
                DIM fp_scaled_score AS INTEGER = FixedMultiply(fp_score, ATTENTION_SCALE_FP) ' Use fixed-point multiplication
                
                ' Store the scaled fixed-point value back as LogQuantized
                scores_mat.data(r, c) = FixedToLogQuantized(fp_scaled_score).packed_value
            NEXT c
        NEXT r
    
        ' Apply causal attention mask (if column > row, set to negative infinity)
        FOR r = 0 TO scores_mat.rows - 1
            FOR c = 0 TO scores_mat.cols - 1
                IF c > r THEN
                    ' Set score to a very large negative fixed-point value
                    scores_mat.data(r, c) = FixedToLogQuantized(NEG_INF_FP).packed_value ' Use fixed-point negative infinity
                END IF
            NEXT c
        NEXT r
    
        ' Apply Softmax to convert scores to normalized probabilities
        MatrixSoftmaxFixed scores_mat, scores_mat
        
        ' Calculate Output (Softmax(Scores) * Value)
        InitMatrix output_mat, scores_mat.rows, value_mat.cols ' Scores.rows = context_length, Value.cols = head_dim
        MatrixMultiply scores_mat, value_mat, output_mat
    
        ' Clean up temporary matrix
        FreeMatrix scores_mat
    END IF
END SUB

' Note on Block-Sparse Attention:
' We've now implemented block-sparse attention in block_sparse.bas and integrated it here.
' The sparse implementation only allocates and computes attention for blocks that are 
' actually needed, which significantly reduces memory usage for longer contexts.
' The block size and decision threshold can be tuned based on performance benchmarks.
' For very short contexts, the overhead of sparse structures may not be worth it.

' Function to implement a single Transformer Attention Head.
' Combines linear projections (Q, K, V weights) and scaled dot-product attention.
' Input: Input matrix (e.g., token embeddings + positional encodings)
' Output: Output matrix for this head
SUB AttentionHead (input_mat AS Matrix, Wq AS Matrix, Wk AS Matrix, Wv AS Matrix, Wo AS Matrix, BYREF output_mat AS Matrix)
    ' Assumes Wq, Wk, Wv are (embedding_dim, head_dim) and Wo is (head_dim, embedding_dim)
    ' Input is (context_length, embedding_dim)

    ' Need temporary matrices for Q, K, V, and intermediate output
    DIM query_mat AS Matrix
    DIM key_mat AS Matrix
    DIM value_mat AS Matrix
    DIM head_output AS Matrix
    InitMatrix query_mat, input_mat.rows, Wq.cols ' (context_length, head_dim)
    InitMatrix key_mat, input_mat.rows, Wk.cols   ' (context_length, head_dim)
    InitMatrix value_mat, input_mat.rows, Wv.cols ' (context_length, head_dim)
    InitMatrix head_output, query_mat.rows, query_mat.cols ' (context_length, head_dim)

    ' Linear Projections: Q = Input * Wq, K = Input * Wk, V = Input * Wv
    ' These need to use the optimized fixed-point MatrixMultiply
    MatrixMultiply input_mat, Wq, query_mat
    MatrixMultiply input_mat, Wk, key_mat
    MatrixMultiply input_mat, Wv, value_mat

    ' Perform Scaled Dot-Product Attention
    ' Note: ScaledDotProductAttention expects Key to be transposed for Q*K^T.
    ' We need to either transpose Key here or modify ScaledDotProductAttention.
    ' Let's assume ScaledDotProductAttention handles the implicit transpose or we add a Transpose function.
    ' For now, calling with Key as is, assuming ScaledDotProductAttention handles it.
    ScaledDotProductAttention query_mat, key_mat, value_mat, head_output

    DIM r AS INTEGER, c AS INTEGER
    IF output_mat.rows <> head_output.rows OR output_mat.cols <> head_output.cols THEN
        FreeMatrix output_mat
        InitMatrix output_mat, head_output.rows, head_output.cols
    END IF

    FOR r = 0 TO head_output.rows - 1
        FOR c = 0 TO head_output.cols - 1
            output_mat.data(r, c) = head_output.data(r, c)
        NEXT c
    NEXT r

    ' Clean up temporary matrices
    FreeMatrix query_mat
    FreeMatrix key_mat
    FreeMatrix value_mat
    FreeMatrix head_output
END SUB

' Function to implement Multi-Head Attention.
' Runs multiple attention heads in parallel and concatenates their outputs.
' Input: Input matrix, Weight matrices for all heads (Wq_all, Wk_all, Wv_all, Wo_all)
' Output: Output matrix
SUB MultiHeadAttention (input_mat AS Matrix, Wq_all AS Matrix, Wk_all AS Matrix, Wv_all AS Matrix, Wo_all AS Matrix, BYREF output_mat AS Matrix)
    ' Assumes Wq_all, Wk_all, Wv_all are (embedding_dim, embedding_dim) and Wo_all is (embedding_dim, embedding_dim)
    ' These weights need to be split and distributed to individual heads.
    ' This requires careful indexing or splitting of the weight matrices.

    ' Need temporary matrices for each head's output
    ' DIM HeadOutputs(NUM_HEADS - 1) AS Matrix ' Array of matrices
    ' For now, let's process heads sequentially and concatenate results.

    DIM runtime_embed_cols AS INTEGER
    DIM runtime_head_count AS INTEGER
    DIM head_width AS INTEGER

    runtime_embed_cols = input_mat.cols
    runtime_head_count = g_config.n_head
    IF runtime_head_count < 1 THEN runtime_head_count = 1
    head_width = runtime_embed_cols \ runtime_head_count
    IF head_width < 1 THEN head_width = 1

    ' Need temporary matrices for weights and output of a single head
    DIM Wq_head AS Matrix: InitMatrix Wq_head, runtime_embed_cols, head_width
    DIM Wk_head AS Matrix: InitMatrix Wk_head, runtime_embed_cols, head_width
    DIM Wv_head AS Matrix: InitMatrix Wv_head, runtime_embed_cols, head_width
    DIM Wo_head AS Matrix: InitMatrix Wo_head, head_width, runtime_embed_cols
    DIM HeadOutput AS Matrix: InitMatrix HeadOutput, input_mat.rows, head_width ' (context_length, head_dim)

    ' Need a temporary matrix to accumulate concatenated head outputs
    DIM ConcatenatedOutput AS Matrix: InitMatrix ConcatenatedOutput, input_mat.rows, runtime_embed_cols ' (context_length, embedding_dim)

    DIM head_idx AS INTEGER
    DIM r AS INTEGER, c AS INTEGER
    FOR head_idx = 0 TO runtime_head_count - 1
        ' Extract weights for the current head from the combined weight matrices
        ExtractHeadWeights Wq_all, Wq_head, head_idx, head_width
        ExtractHeadWeights Wk_all, Wk_head, head_idx, head_width
        ExtractHeadWeights Wv_all, Wv_head, head_idx, head_width

        ' Perform attention for this head
        AttentionHead input_mat, Wq_head, Wk_head, Wv_head, Wo_head, HeadOutput

        ' Concatenate the output of this head into the ConcatenatedOutput matrix
        ConcatenateHeadOutput ConcatenatedOutput, HeadOutput, head_idx, head_width

    NEXT head_idx

    ' Final linear projection after concatenation (Wo_all)
    ' The Wo_all matrix is applied to the concatenated output.
    ' Need a temporary matrix for the final output
    DIM FinalOutput AS Matrix: InitMatrix FinalOutput, ConcatenatedOutput.rows, Wo_all.cols ' (context_length, embedding_dim)
    MatrixMultiply ConcatenatedOutput, Wo_all, FinalOutput

    ' Copy result to Output matrix (assuming Output is pre-initialized)
    FOR r = 0 TO FinalOutput.rows - 1
        FOR c = 0 TO FinalOutput.cols - 1
            output_mat.data(r, c) = FinalOutput.data(r, c)
        NEXT c
    NEXT r

    ' Clean up temporary matrices
    FreeMatrix Wq_head
    FreeMatrix Wk_head
    FreeMatrix Wv_head
    FreeMatrix Wo_head
    FreeMatrix HeadOutput
    FreeMatrix ConcatenatedOutput
    FreeMatrix FinalOutput
END SUB

' Function to implement a Feed-Forward Network (FFN) block with GLU.
' Input: Input matrix
' Output: Output matrix
SUB FeedForward (input_mat AS Matrix, W1 AS Matrix, W2 AS Matrix, W3 AS Matrix, BYREF output_mat AS Matrix)
    ' Assumes W1, W3 are (embedding_dim, intermediate_dim) and W2 is (intermediate_dim, embedding_dim)
    ' intermediate_dim is typically 4 * embedding_dim
    ' GLU involves two linear layers (Input * W1 and Input * W3) and element-wise multiplication after a non-linearity on one.

    ' Need temporary matrices
    DIM Intermediate1 AS Matrix: InitMatrix Intermediate1, input_mat.rows, W1.cols ' (context_length, intermediate_dim)
    DIM Intermediate2 AS Matrix: InitMatrix Intermediate2, input_mat.rows, W3.cols ' (context_length, intermediate_dim)
    DIM ActivatedIntermediate AS Matrix: InitMatrix ActivatedIntermediate, input_mat.rows, W1.cols ' (context_length, intermediate_dim)

    ' First linear layer: Intermediate1 = Input * W1
    MatrixMultiply input_mat, W1, Intermediate1

    ' Second linear layer for GLU gate: Intermediate2 = Input * W3
    MatrixMultiply input_mat, W3, Intermediate2

    ' Apply GELU activation - using the optimized implementation
    ' GELU is the activation function used in modern transformer models like GPT-2
    ApplyGELU Intermediate1, ActivatedIntermediate

    ' Apply GLU gate: ActivatedIntermediate = ActivatedIntermediate * Intermediate2 (element-wise multiplication)
    ' Use the fixed-point element-wise multiplication function.
    MatrixElementWiseMultiply ActivatedIntermediate, Intermediate2, ActivatedIntermediate ' Overwrite ActivatedIntermediate

    ' Third linear layer: Output = ActivatedIntermediate * W2
    MatrixMultiply ActivatedIntermediate, W2, output_mat

    ' Clean up temporary matrices
    FreeMatrix Intermediate1
    FreeMatrix Intermediate2
    FreeMatrix ActivatedIntermediate
END SUB

' Function to implement Layer Normalization.
' Input: Input matrix
' Output: Output matrix
' Assumes gamma and beta parameters are stored elsewhere (as LogQuantized matrices).
SUB LayerNorm (input_mat AS Matrix, Gamma AS Matrix, Beta AS Matrix, BYREF output_mat AS Matrix)
    ' Assumes Input, Gamma, Beta, Output are (context_length, embedding_dim)
    ' LayerNorm normalizes across the embedding dimension for each token (row).
    ' Calculation: (x - mean) / sqrt(variance + epsilon) * gamma + beta
    ' This requires calculating mean and variance for each row, fixed-point division,
    ' square root, and element-wise operations.

    ' Need temporary matrix for normalized output
    DIM NormalizedInput AS Matrix: InitMatrix NormalizedInput, input_mat.rows, input_mat.cols

    DIM r AS INTEGER ' Row index (token)
    DIM c AS INTEGER ' Column index (embedding dimension)
    DIM fp_sum AS INTEGER
    DIM fp_mean AS INTEGER
    DIM fp_sum_sq_diff AS INTEGER
    DIM fp_variance AS INTEGER
    DIM fp_stddev AS INTEGER
    DIM fp_inv_stddev AS INTEGER
    DIM fp_val AS INTEGER
    DIM fp_diff AS INTEGER
    DIM fp_sq_diff AS INTEGER
    DIM fp_normalized_val AS INTEGER

    ' Define fixed-point epsilon for numerical stability
    CONST EPSILON_FP AS INTEGER = 1 ' Approximately 1e-5 in 16.16 fixed point

    FOR r = 0 TO input_mat.rows - 1
        ' Calculate mean for the current row (token) in fixed-point
        fp_sum = 0
        FOR c = 0 TO input_mat.cols - 1
            fp_sum = FixedAdd(fp_sum, DequantizeLogToFixed(input_mat.data(r, c))) ' Use direct dequantization
        NEXT c
        ' Need fixed-point division by Input.cols
        fp_mean = FixedDivide(fp_sum, FloatToFixed(CSNG(input_mat.cols))) ' Use FixedDivide

        ' Calculate variance for the current row (token) in fixed-point
        fp_sum_sq_diff = 0
        FOR c = 0 TO input_mat.cols - 1
            fp_val = DequantizeLogToFixed(input_mat.data(r, c)) ' Use direct dequantization
            fp_diff = FixedSubtract(fp_val, fp_mean) ' Use FixedSubtract
            fp_sq_diff = FixedMultiply(fp_diff, fp_diff) ' Use FixedMultiply
            fp_sum_sq_diff = FixedAdd(fp_sum_sq_diff, fp_sq_diff) ' Use FixedAdd
        NEXT c
        ' Need fixed-point division by Input.cols
        fp_variance = FixedDivide(fp_sum_sq_diff, FloatToFixed(CSNG(input_mat.cols))) ' Use FixedDivide

        ' Add epsilon for numerical stability (fixed-point epsilon)
        fp_variance = FixedAdd(fp_variance, EPSILON_FP) ' Use FixedAdd

        ' Calculate inverse standard deviation (1 / sqrt(variance + epsilon))
        ' Need fixed-point square root and division.
        ' Use FixedSqrt and FixedDivide.
        fp_stddev = FixedSqrt(fp_variance) ' Use FixedSqrt
        fp_inv_stddev = FixedDivide(FloatToFixed(1.0), fp_stddev) ' Use FixedDivide

        ' Normalize the row: (x - mean) * inv_stddev
        FOR c = 0 TO input_mat.cols - 1
            fp_val = DequantizeLogToFixed(input_mat.data(r, c)) ' Use direct dequantization
            fp_diff = FixedSubtract(fp_val, fp_mean) ' Use FixedSubtract
            fp_normalized_val = FixedMultiply(fp_diff, fp_inv_stddev) ' Use FixedMultiply
            NormalizedInput.data(r, c) = FixedToLogQuantized(fp_normalized_val).packed_value ' Convert back to LogQuantized
        NEXT c
    NEXT r

    ' Apply learned parameters gamma and beta (element-wise)
    ' Output = NormalizedInput * gamma + beta
    ' Use fixed-point element-wise multiplication and addition.
    
    DIM gamma_fp AS INTEGER, beta_fp AS INTEGER
    DIM scaled_fp AS INTEGER

    IF output_mat.rows <> input_mat.rows OR output_mat.cols <> input_mat.cols THEN
        FreeMatrix output_mat
        InitMatrix output_mat, input_mat.rows, input_mat.cols
    END IF

    FOR r = 0 TO NormalizedInput.rows - 1
        FOR c = 0 TO NormalizedInput.cols - 1
            gamma_fp = FloatToFixed(1.0)
            beta_fp = 0

            IF Gamma.rows = NormalizedInput.cols AND Gamma.cols = 1 THEN
                gamma_fp = DequantizeLogToFixed(Gamma.data(c, 0))
            ELSEIF Gamma.rows = NormalizedInput.rows AND Gamma.cols = NormalizedInput.cols THEN
                gamma_fp = DequantizeLogToFixed(Gamma.data(r, c))
            END IF

            IF Beta.rows = NormalizedInput.cols AND Beta.cols = 1 THEN
                beta_fp = DequantizeLogToFixed(Beta.data(c, 0))
            ELSEIF Beta.rows = NormalizedInput.rows AND Beta.cols = NormalizedInput.cols THEN
                beta_fp = DequantizeLogToFixed(Beta.data(r, c))
            END IF

            scaled_fp = FixedAdd(FixedMultiply(DequantizeLogToFixed(NormalizedInput.data(r, c)), gamma_fp), beta_fp)
            output_mat.data(r, c) = FixedToLogQuantized(scaled_fp).packed_value
        NEXT c
    NEXT r

    ' Clean up temporary matrices
    FreeMatrix NormalizedInput
END SUB

' Note on Layer Normalization Optimization:
' Calculating mean and variance row-wise involves sums and divisions.
' Fixed-point division and square root are now implemented using FixedDivide and FixedSqrt.
' Approximations or lookup tables for these operations might still be necessary for
' optimal performance on 486, depending on the efficiency of the FixedSqrt implementation.
' The element-wise multiplication and addition with gamma and beta are handled
' with fixed-point arithmetic.

' *******************************************************
' * Helper Functions for Transformers                   *
' *******************************************************

' Extract weights for a specific attention head from combined weights matrix
SUB ExtractHeadWeights(W_all AS Matrix, BYREF W_head AS Matrix, head_idx AS INTEGER, head_width AS INTEGER)
    DIM start_col AS INTEGER
    DIM i AS INTEGER, j AS INTEGER
    
    ' Calculate the starting column for this head's weights
    start_col = head_idx * head_width
    
    ' Ensure W_head is properly dimensioned
    IF W_head.rows <> W_all.rows OR W_head.cols <> head_width THEN
        FreeMatrix(W_head)
        InitMatrix(W_head, W_all.rows, head_width)
    END IF
    
    ' Copy the appropriate slice of weights
    FOR i = 0 TO W_all.rows - 1
        FOR j = 0 TO head_width - 1
            W_head.data(i, j) = W_all.data(i, start_col + j)
        NEXT j
    NEXT i
END SUB

' Concatenate a head's output into the combined output matrix
SUB ConcatenateHeadOutput(BYREF Concatenated AS Matrix, HeadOutput AS Matrix, head_idx AS INTEGER, head_width AS INTEGER)
    DIM start_col AS INTEGER
    DIM i AS INTEGER, j AS INTEGER
    
    ' Calculate the starting column for this head's output
    start_col = head_idx * head_width
    
    ' Copy HeadOutput to the correct slice of ConcatenatedOutput
    FOR i = 0 TO HeadOutput.rows - 1
        FOR j = 0 TO head_width - 1
            Concatenated.data(i, start_col + j) = HeadOutput.data(i, j)
        NEXT j
    NEXT i
END SUB

' Apply GELU activation function element-wise for transformer feed-forward networks
' Using the accurate approximation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
SUB ApplyGELU(A AS Matrix, BYREF B AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    DIM sqrt_2_div_pi_fp AS INTEGER
    DIM coef_fp AS INTEGER
    DIM x_fp AS INTEGER, x3_fp AS INTEGER, inner_fp AS INTEGER, tanh_arg_fp AS INTEGER
    DIM tanh_result_fp AS INTEGER, result_fp AS INTEGER
    
    ' Constants for GELU approximation in fixed-point
    sqrt_2_div_pi_fp = FloatToFixed(0.7978845608) ' sqrt(2/π)
    coef_fp = FloatToFixed(0.044715)
    
    ' Initialize result matrix if needed
    IF B.rows <> A.rows OR B.cols <> A.cols THEN
        FreeMatrix(B)
        InitMatrix(B, A.rows, A.cols)
    END IF
    
    ' Apply GELU to each element using fixed-point arithmetic
    FOR i = 0 TO A.rows - 1
        FOR j = 0 TO A.cols - 1
            ' Get value in fixed-point
            x_fp = DequantizeLogToFixed(A.data(i, j))
            
            ' Calculate x^3
            x3_fp = FixedMultiply(x_fp, FixedMultiply(x_fp, x_fp))
            
            ' Calculate 0.044715 * x^3
            x3_fp = FixedMultiply(coef_fp, x3_fp)
            
            ' Calculate x + 0.044715 * x^3
            inner_fp = FixedAdd(x_fp, x3_fp)
            
            ' Calculate sqrt(2/π) * (x + 0.044715 * x^3)
            tanh_arg_fp = FixedMultiply(sqrt_2_div_pi_fp, inner_fp)
            
            ' Calculate tanh of the argument using approximation
            ' tanh(x) ≈ x / (1 + |x|) for small x, more complex for larger x
            ' Here we'll use a simple approximation
            IF tanh_arg_fp < -FloatToFixed(3.0) THEN
                tanh_result_fp = FloatToFixed(-1.0) ' saturate for large negative
            ELSEIF tanh_arg_fp > FloatToFixed(3.0) THEN
                tanh_result_fp = FloatToFixed(1.0) ' saturate for large positive
            ELSE
                ' x / (1 + |x|) approximation for small x
                DIM abs_x_fp AS INTEGER = ABS(tanh_arg_fp)
                DIM denom_fp AS INTEGER = FixedAdd(FloatToFixed(1.0), abs_x_fp)
                tanh_result_fp = FixedDivide(tanh_arg_fp, denom_fp)
            END IF
            
            ' Calculate 1 + tanh(...)
            tanh_result_fp = FixedAdd(FloatToFixed(1.0), tanh_result_fp)
            
            ' Calculate x * (1 + tanh(...))
            result_fp = FixedMultiply(x_fp, tanh_result_fp)
            
            ' Calculate 0.5 * x * (1 + tanh(...))
            result_fp = FixedMultiply(FloatToFixed(0.5), result_fp)
            
            ' Store result
            B.data(i, j) = FixedToLogQuantized(result_fp).packed_value
        NEXT j
    NEXT i
END SUB
