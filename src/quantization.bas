' *******************************************************
' * Logarithmic Quantization Implementation for GPT-2   *
' *******************************************************
' * This file contains the 4-bit logarithmic quantization
' * scheme used to compress model weights and activations.
' * Uses fixed-point arithmetic for 486 compatibility.
' *******************************************************

#INCLUDE "matrix_ops.bas" ' For fixed-point types and operations

' *******************************************************
' * Type Definitions and Constants                       *
' *******************************************************

' Define the 4-bit logarithmic quantized value type
TYPE LogQuantized
    packed_value AS INTEGER ' Contains sign, 4-bit mantissa and 4-bit exponent
END TYPE

' Constants for bit operations
CONST MANTISSA_BITS = 4
CONST EXPONENT_BITS = 4
CONST MANTISSA_MASK = &HF       ' 0000 1111
CONST EXPONENT_MASK = &HF0      ' 1111 0000
CONST EXPONENT_SHIFT = 4
CONST EXPONENT_BIAS = 8

' Lookup tables for efficient conversion
DIM SHARED DequantFixedLookup(0 TO 510) AS INTEGER ' Fixed-point values for dequantization
DIM SHARED QuantFixedLookup(-32768 TO 32767) AS INTEGER ' Quantized values for fixed-point inputs

' *******************************************************
' * Initialization Functions                             *
' *******************************************************

' Initialize lookup tables for efficient quantization/dequantization
SUB InitQuantizationTables()
    DIM i AS INTEGER, packed_val AS INTEGER, fixed_val AS INTEGER
    DIM lookup_index AS INTEGER, sgn AS INTEGER, abs_packed_val AS INTEGER
    DIM mantissa AS INTEGER, exponent AS INTEGER
    DIM fp_power_of_2 AS INTEGER, fp_mantissa_scale AS INTEGER, fp_result AS INTEGER
    DIM fixed_min AS INTEGER, fixed_max AS INTEGER, bucket_size AS INTEGER
    
    PRINT "Initializing quantization lookup tables..."
    
    ' Initialize dequantization lookup table (packed -> fixed-point)
    FOR packed_val = -255 TO 255
        lookup_index = packed_val + 255
        
        ' Handle the special case for zero
        IF packed_val = 0 THEN
            DequantFixedLookup(lookup_index) = 0
        ELSE
            ' Determine the sign
            IF packed_val > 0 THEN sgn = 1 ELSE sgn = -1
            
            ' Get the absolute packed value
            abs_packed_val = ABS(packed_val)
            
            ' Extract mantissa and exponent from the absolute packed value
            mantissa = abs_packed_val AND MANTISSA_MASK
            exponent = (abs_packed_val AND EXPONENT_MASK) >> EXPONENT_SHIFT
            
            ' Calculate 2^(exponent-bias) in fixed-point
            fp_power_of_2 = FixedPow2(exponent - EXPONENT_BIAS)
            
            ' Calculate mantissa/16 in fixed-point
            fp_mantissa_scale = FixedDivide(FloatToFixed(mantissa), FloatToFixed(16.0))
            
            ' Calculate final result: (mantissa/16) * 2^(exponent-bias)
            fp_result = FixedMultiply(fp_mantissa_scale, fp_power_of_2)
            
            ' Apply sign
            IF sgn = -1 THEN fp_result = -fp_result
            
            ' Store in lookup table
            DequantFixedLookup(lookup_index) = fp_result
        END IF
    NEXT packed_val
    
    ' Initialize quantization lookup table (fixed-point -> packed)
    ' Create buckets that map ranges of fixed-point values to quantized values
    
    ' Determine the range of fixed-point values to cover
    fixed_min = FloatToFixed(-8.0)  ' Lower bound for quantization
    fixed_max = FloatToFixed(8.0)   ' Upper bound for quantization
    
    ' First, set all values to zero
    FOR i = -32768 TO 32767
        QuantFixedLookup(i) = 0
    NEXT i
    
    ' Fill lookup table by iterating through all possible packed values
    ' and mapping each fixed-point value to the closest packed value
    FOR packed_val = -255 TO 255
        ' Skip zero (already set)
        IF packed_val = 0 THEN CONTINUE FOR
        
        lookup_index = packed_val + 255
        fixed_val = DequantFixedLookup(lookup_index)
        
        ' Find next packed value to determine the range
        DIM next_packed_val AS INTEGER, next_fixed_val AS INTEGER
        DIM prev_packed_val AS INTEGER, prev_fixed_val AS INTEGER
        
        ' Handle edge cases
        IF packed_val = 255 THEN
            next_packed_val = packed_val
            next_fixed_val = FloatToFixed(100.0) ' Large positive number
        ELSE
            next_packed_val = packed_val + 1
            next_fixed_val = DequantFixedLookup(next_packed_val + 255)
        END IF
        
        IF packed_val = -255 THEN
            prev_packed_val = packed_val
            prev_fixed_val = FloatToFixed(-100.0) ' Large negative number
        ELSE
            prev_packed_val = packed_val - 1
            prev_fixed_val = DequantFixedLookup(prev_packed_val + 255)
        END IF
        
        ' Calculate midpoints for bucketing
        DIM lower_bound AS INTEGER, upper_bound AS INTEGER
        
        ' For positive values
        IF fixed_val > 0 THEN
            lower_bound = (fixed_val + prev_fixed_val) / 2
            upper_bound = (fixed_val + next_fixed_val) / 2
            
            ' Handle edge cases
            IF lower_bound < 0 THEN lower_bound = 0
            
            ' Fill the bucket
            FOR i = lower_bound TO upper_bound
                IF i >= -32768 AND i <= 32767 THEN ' Ensure within table bounds
                    QuantFixedLookup(i) = packed_val
                END IF
            NEXT i
        ' For negative values
        ELSEIF fixed_val < 0 THEN
            lower_bound = (fixed_val + next_fixed_val) / 2
            upper_bound = (fixed_val + prev_fixed_val) / 2
            
            ' Handle edge cases
            IF upper_bound > 0 THEN upper_bound = 0
            
            ' Fill the bucket
            FOR i = lower_bound TO upper_bound
                IF i >= -32768 AND i <= 32767 THEN ' Ensure within table bounds
                    QuantFixedLookup(i) = packed_val
                END IF
            NEXT i
        END IF
    NEXT packed_val
    
    PRINT "Quantization tables initialized."
END SUB

' Fixed-point power of 2 function: calculates 2^n in fixed-point
FUNCTION FixedPow2(n AS INTEGER) AS INTEGER
    ' For positive exponents: left shift
    IF n >= 0 THEN
        ' Handle overflow protection
        IF n > 15 THEN
            RETURN &H7FFFFFFF ' Maximum INTEGER value
        ELSE
            RETURN FIXED_POINT_ONE << n
        END IF
    ' For negative exponents: right shift
    ELSE
        ' Handle underflow protection
        IF n < -15 THEN
            RETURN 0
        ELSE
            RETURN FIXED_POINT_ONE >> ABS(n)
        END IF
    END IF
END FUNCTION

' *******************************************************
' * Quantization Functions                               *
' *******************************************************

' Quantize a floating-point value to 4-bit log representation
FUNCTION QuantizeLog(f AS SINGLE) AS LogQuantized
    DIM lq AS LogQuantized
    DIM fp_val AS INTEGER
    
    ' Convert float to fixed-point, then quantize
    fp_val = FloatToFixed(f)
    lq = FixedToLogQuantized(fp_val)
    
    FUNCTION = lq
END FUNCTION

' Quantize a fixed-point value to LogQuantized using lookup table
FUNCTION FixedToLogQuantized(fp AS INTEGER) AS LogQuantized
    DIM lq AS LogQuantized
    
    ' Handle out-of-range values
    IF fp < -32768 THEN
        fp = -32768
    ELSEIF fp > 32767 THEN
        fp = 32767
    END IF
    
    ' Use lookup table for direct mapping
    lq.packed_value = QuantFixedLookup(fp)
    
    FUNCTION = lq
END FUNCTION

' Fixed-point log2 approximation (used when initializing lookup tables)
FUNCTION FixedLog2(x AS INTEGER) AS INTEGER
    ' Implementation for log2 approximation in fixed-point
    ' using Taylor series or lookup tables
    
    ' Handle special cases
    IF x <= 0 THEN RETURN FloatToFixed(-16.0) ' Large negative for log(0) or negative
    IF x = FIXED_POINT_ONE THEN RETURN 0 ' log2(1) = 0
    
    ' Find the position of the highest bit to get integer part of log2
    DIM bit_pos AS INTEGER = 31
    DIM int_part AS INTEGER = 0
    DIM val AS INTEGER = x
    
    ' Find most significant bit
    WHILE bit_pos >= 0
        IF (val AND (1 << bit_pos)) <> 0 THEN
            int_part = bit_pos - FIXED_POINT_SHIFT
            EXIT WHILE
        END IF
        bit_pos = bit_pos - 1
    WEND
    
    ' Normalize x to range [1, 2) in fixed-point
    val = val >> int_part
    
    ' Use table or polynomial approximation for fractional part
    ' For simplicity, using a first-order approximation
    DIM frac_part AS INTEGER = FixedMultiply(FloatToFixed(0.5), (val - FIXED_POINT_ONE))
    
    ' Combine integer and fractional parts
    RETURN FloatToFixed(CSNG(int_part)) + frac_part
END FUNCTION

' *******************************************************
' * Dequantization Functions                            *
' *******************************************************

' Dequantize a LogQuantized value to SINGLE (useful for debugging)
FUNCTION DequantizeLog(lq AS LogQuantized) AS SINGLE
    DIM fp_val AS INTEGER
    
    ' Convert to fixed-point first
    fp_val = LogQuantizedToFixed(lq)
    
    ' Then convert fixed-point to float
    FUNCTION = FixedToFloat(fp_val)
END FUNCTION

' Dequantize a LogQuantized value directly to fixed-point
FUNCTION LogQuantizedToFixed(lq AS LogQuantized) AS INTEGER
    DIM packed_val AS INTEGER = lq.packed_value
    
    ' Use lookup table for fast dequantization
    DIM lookup_index AS INTEGER = packed_val + 255
    
    ' Bounds checking
    IF lookup_index < 0 THEN lookup_index = 0
    IF lookup_index > 510 THEN lookup_index = 510
    
    FUNCTION = DequantFixedLookup(lookup_index)
END FUNCTION

' *******************************************************
' * Matrix Quantization Functions                       *
' *******************************************************

' Quantize a matrix of fixed-point values to LogQuantized format
SUB QuantizeMatrixToLog(A AS Matrix, BYREF B AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    
    ' Initialize output matrix if needed
    IF B.rows <> A.rows OR B.cols <> A.cols THEN
        FreeMatrix(B)
        InitMatrix(B, A.rows, A.cols)
    END IF
    
    ' Quantize each element
    FOR i = 0 TO A.rows - 1
        FOR j = 0 TO A.cols - 1
            B.data(i, j) = FixedToLogQuantized(A.data(i, j)).packed_value
        NEXT j
    NEXT i
END SUB

' Dequantize a matrix of LogQuantized values to fixed-point
SUB DequantizeMatrixFromLog(A AS Matrix, BYREF B AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    DIM lq AS LogQuantized
    
    ' Initialize output matrix if needed
    IF B.rows <> A.rows OR B.cols <> A.cols THEN
        FreeMatrix(B)
        InitMatrix(B, A.rows, A.cols)
    END IF
    
    ' Dequantize each element
    FOR i = 0 TO A.rows - 1
        FOR j = 0 TO A.cols - 1
            lq.packed_value = A.data(i, j)
            B.data(i, j) = LogQuantizedToFixed(lq)
        NEXT j
    NEXT i
END SUB

' *******************************************************
' * Testing Functions                                   *
' *******************************************************

' Test the quantization accuracy
SUB TestQuantization()
    DIM i AS INTEGER
    DIM f AS SINGLE, fp AS INTEGER, re_fp AS INTEGER
    DIM lq AS LogQuantized
    DIM max_error AS SINGLE, avg_error AS SINGLE
    
    PRINT "Testing LogQuantized Quantization..."
    PRINT "-----------------------------------"
    
    ' Initialize lookup tables
    InitQuantizationTables()
    
    ' Test a range of values
    max_error = 0.0
    avg_error = 0.0
    
    FOR i = -20 TO 20
        ' Test value (nonlinearly spaced for better coverage)
        f = i * 0.1
        IF i <> 0 THEN f = f * ABS(f)
        
        ' Convert to fixed-point
        fp = FloatToFixed(f)
        
        ' Quantize and dequantize
        lq = FixedToLogQuantized(fp)
        re_fp = LogQuantizedToFixed(lq)
        
        ' Calculate error
        DIM error AS SINGLE
        error = ABS(FixedToFloat(fp) - FixedToFloat(re_fp))
        
        ' Update statistics
        avg_error = avg_error + error
        IF error > max_error THEN max_error = error
        
        ' Print select results
        IF i MOD 5 = 0 THEN
            PRINT "Original: "; f; " Fixed: "; FixedToFloat(fp); _
                  " Packed: "; lq.packed_value; " Reconstructed: "; FixedToFloat(re_fp); _
                  " Error: "; error
        END IF
    NEXT i
    
    avg_error = avg_error / 41
    
    PRINT
    PRINT "Quantization Results:"
    PRINT "  Maximum error: "; max_error
    PRINT "  Average error: "; avg_error
    PRINT
    
    ' Test a matrix
    DIM test_mat AS Matrix, quant_mat AS Matrix, dequant_mat AS Matrix
    InitMatrix(test_mat, 4, 4)
    
    ' Fill with test values
    FOR i = 0 TO test_mat.rows - 1
        FOR j = 0 TO test_mat.cols - 1
            test_mat.data(i, j) = FloatToFixed(i * 0.1 - j * 0.2)
        NEXT j
    NEXT i
    
    ' Quantize and dequantize
    QuantizeMatrixToLog(test_mat, quant_mat)
    DequantizeMatrixFromLog(quant_mat, dequant_mat)
    
    ' Calculate error
    max_error = 0.0
    avg_error = 0.0
    
    FOR i = 0 TO test_mat.rows - 1
        FOR j = 0 TO test_mat.cols - 1
            error = ABS(FixedToFloat(test_mat.data(i, j)) - FixedToFloat(dequant_mat.data(i, j)))
            avg_error = avg_error + error
            IF error > max_error THEN max_error = error
        NEXT j
    NEXT i
    
    avg_error = avg_error / (test_mat.rows * test_mat.cols)
    
    PRINT "Matrix Quantization Results:"
    PRINT "  Maximum error: "; max_error
    PRINT "  Average error: "; avg_error
    
    ' Clean up
    FreeMatrix(test_mat)
    FreeMatrix(quant_mat)
    FreeMatrix(dequant_mat)
END SUB
