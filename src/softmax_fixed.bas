' Fixed-point Softmax implementation for the GPT-2-like model.
' This file provides an efficient implementation of the Softmax function
' using fixed-point arithmetic, optimized for 486-era constraints.

' Include necessary files
#INCLUDE "data_structures.bas"
#INCLUDE "quantization.bas" ' For LogQuantized type and dequantization to fixed-point
#INCLUDE "matrix_ops.bas" ' For matrix operations and fixed-point arithmetic

' Constants for exponential approximation
' Using Q16.16 fixed-point representation (same as in matrix_ops.bas)
CONST EXP_TABLE_SIZE AS INTEGER = 1024 ' Size of the exponential lookup table
CONST EXP_TABLE_RANGE AS INTEGER = 10 ' Range of input values for exp table (0 to 10)
CONST EXP_FIXED_SCALE AS INTEGER = EXP_TABLE_SIZE / EXP_TABLE_RANGE ' Scale factor for lookup
DIM ExpLookupTable(0 TO EXP_TABLE_SIZE - 1) AS INTEGER ' Fixed-point exp(x) values

' Function to initialize the exponential lookup table.
' This pre-computes exp(x) values for the range [0, EXP_TABLE_RANGE)
SUB InitExpLookupTable()
    DIM i AS INTEGER
    DIM x AS SINGLE
    FOR i = 0 TO EXP_TABLE_SIZE - 1
        x = (CSNG(i) * EXP_TABLE_RANGE) / EXP_TABLE_SIZE
        ExpLookupTable(i) = FloatToFixed(EXP(x))
    NEXT i
    PRINT "Exponential lookup table initialized."
END SUB

' Function to approximate exp(x) using the lookup table
' Input: x in fixed-point (Q16.16)
' Output: exp(x) in fixed-point (Q16.16)
FUNCTION FixedExp (fp_x AS INTEGER) AS INTEGER
    ' Handle negative values (exp(-x) = 1/exp(x))
    IF fp_x < 0 THEN
        ' Compute exp(-x) and then take reciprocal
        DIM neg_x AS INTEGER = -fp_x
        DIM fp_exp_neg_x AS INTEGER = FixedExp(neg_x)
        ' Return 1/exp(-x) using fixed-point division
        FUNCTION = FixedDivide(FIXED_POINT_SCALE, fp_exp_neg_x)
        EXIT FUNCTION
    END IF
    
    ' Handle input range - anything above EXP_TABLE_RANGE will likely overflow
    ' Cap inputs to prevent issues
    IF fp_x > FloatToFixed(EXP_TABLE_RANGE) THEN
        fp_x = FloatToFixed(EXP_TABLE_RANGE)
    END IF
    
    ' Convert fixed-point value to table index
    DIM index AS INTEGER = CLNG(CLNG(fp_x) * EXP_TABLE_SIZE) \ (EXP_TABLE_RANGE * FIXED_POINT_SCALE)
    
    ' Ensure index is in range
    IF index < 0 THEN index = 0
    IF index >= EXP_TABLE_SIZE THEN index = EXP_TABLE_SIZE - 1
    
    ' Return the exponential from lookup table
    FUNCTION = ExpLookupTable(index)
END FUNCTION

' Function to apply Softmax to a matrix row-wise
' Input: Matrix of logits (context_length, vocab_size)
' Output: Same matrix with Softmax probabilities
SUB SoftmaxFixedPoint (Matrix AS Matrix)
    DIM r AS INTEGER ' Row index
    DIM c AS INTEGER ' Column index
    
    FOR r = 0 TO Matrix.rows - 1
        ' Step 1: Find the maximum value in the row for numerical stability
        DIM max_val AS INTEGER = -2147483647 ' Minimum INT value
        
        FOR c = 0 TO Matrix.cols - 1
            DIM fp_val AS INTEGER = LogQuantizedToFixed(Matrix.data(r, c))
            IF fp_val > max_val THEN max_val = fp_val
        NEXT c
        
        ' Step 2: Compute exp(x - max) for each element in the row and sum them
        DIM fp_sum AS INTEGER = 0 ' Accumulate the sum of exponentials
        DIM exp_values(0 TO Matrix.cols - 1) AS INTEGER ' Store exp values temporarily
        
        FOR c = 0 TO Matrix.cols - 1
            DIM fp_val AS INTEGER = LogQuantizedToFixed(Matrix.data(r, c))
            DIM fp_shifted AS INTEGER = FixedSubtract(fp_val, max_val) ' x - max
            
            ' Compute exp(x - max) for numerical stability
            exp_values(c) = FixedExp(fp_shifted)
            
            ' Accumulate sum for normalization
            fp_sum = FixedAdd(fp_sum, exp_values(c))
        NEXT c
        
        ' Step 3: Normalize by dividing each exp value by the sum
        FOR c = 0 TO Matrix.cols - 1
            ' Skip division if sum is zero (numerical underflow protection)
            DIM fp_prob AS INTEGER
            IF fp_sum > 0 THEN
                fp_prob = FixedDivide(exp_values(c), fp_sum)
            ELSE
                ' If sum is zero, distribute probability uniformly
                fp_prob = FixedDivide(FIXED_POINT_SCALE, FloatToFixed(CSNG(Matrix.cols)))
            END IF
            
            ' Store the normalized probability as LogQuantized
            Matrix.data(r, c) = FixedToLogQuantized(fp_prob).packed_value
        NEXT c
    NEXT r
END SUB

' Optimized version of Softmax for a single row (vector)
' This is useful for sampling from probability distributions
SUB SoftmaxVectorFixedPoint (Vector AS Matrix)
    ' Assumes Vector is a matrix with dimensions (1, size)
    DIM c AS INTEGER ' Column index
    
    ' Step 1: Find the maximum value for numerical stability
    DIM max_val AS INTEGER = -2147483647 ' Minimum INT value
    
    FOR c = 0 TO Vector.cols - 1
        DIM fp_val AS INTEGER = LogQuantizedToFixed(Vector.data(0, c))
        IF fp_val > max_val THEN max_val = fp_val
    NEXT c
    
    ' Step 2: Compute exp(x - max) for each element and sum them
    DIM fp_sum AS INTEGER = 0 ' Accumulate the sum of exponentials
    DIM exp_values(0 TO Vector.cols - 1) AS INTEGER ' Store exp values temporarily
    
    FOR c = 0 TO Vector.cols - 1
        DIM fp_val AS INTEGER = LogQuantizedToFixed(Vector.data(0, c))
        DIM fp_shifted AS INTEGER = FixedSubtract(fp_val, max_val) ' x - max
        
        ' Compute exp(x - max) for numerical stability
        exp_values(c) = FixedExp(fp_shifted)
        
        ' Accumulate sum for normalization
        fp_sum = FixedAdd(fp_sum, exp_values(c))
    NEXT c
    
    ' Step 3: Normalize by dividing each exp value by the sum
    FOR c = 0 TO Vector.cols - 1
        ' Skip division if sum is zero (numerical underflow protection)
        DIM fp_prob AS INTEGER
        IF fp_sum > 0 THEN
            fp_prob = FixedDivide(exp_values(c), fp_sum)
        ELSE
            ' If sum is zero, distribute probability uniformly
            fp_prob = FixedDivide(FIXED_POINT_SCALE, FloatToFixed(CSNG(Vector.cols)))
        END IF
        
        ' Store the normalized probability as LogQuantized
        Vector.data(0, c) = FixedToLogQuantized(fp_prob).packed_value
    NEXT c
END SUB
