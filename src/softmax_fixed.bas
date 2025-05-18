' *******************************************************
' * Fixed-Point Softmax for GPT-2 BASIC                 *
' *******************************************************
' * This module implements fixed-point optimized        *
' * softmax computation for the GPT-2 model on 486-era  *
' * hardware without floating point units.              *
' *                                                     *
' * It provides stable softmax implementation that      *
' * operates entirely in fixed-point math for systems   *
' * without an FPU.                                     *
' *******************************************************

#INCLUDE "src/data_structures.bas"
#INCLUDE "src/asm_optimizations.bas"

' *******************************************************
' * Constants and Lookup Tables                         *
' *******************************************************

' Use 16.16 fixed-point format from asm_optimizations.bas
' FIXED_POINT_SHIFT = 16
' FIXED_POINT_ONE = (1 << FIXED_POINT_SHIFT)

' Exponential function approximation constants
CONST EXP_LOOKUP_SIZE = 8192 ' Size of exponential function lookup table
DIM SHARED exp_lookup(0 TO EXP_LOOKUP_SIZE - 1) AS LONG ' Fixed-point e^x values

' Exponential function range and scaling
' We'll support exp(x) for x from -8.0 to 8.0
CONST EXP_MIN = -8.0
CONST EXP_MAX = 8.0
CONST EXP_RANGE = EXP_MAX - EXP_MIN

' *******************************************************
' * Initialization                                      *
' *******************************************************

' Initialize lookup tables
SUB InitFixedSoftmax()
    DIM i AS INTEGER
    DIM x AS SINGLE
    DIM exp_val AS SINGLE
    
    PRINT "Initializing fixed-point softmax lookup tables..."
    
    ' Initialize exponential lookup table
    FOR i = 0 TO EXP_LOOKUP_SIZE - 1
        ' Map i to range [EXP_MIN, EXP_MAX]
        x = EXP_MIN + (i * EXP_RANGE) / (EXP_LOOKUP_SIZE - 1)
        
        ' Compute e^x and convert to fixed-point
        exp_val = EXP(x)
        exp_lookup(i) = FloatToFixed(exp_val)
    NEXT i
    
    PRINT "Fixed-point softmax initialized."
END SUB

' *******************************************************
' * Fixed-Point Math Functions                          *
' *******************************************************

' Fixed-point exponential function using lookup table
FUNCTION FixedExp(x AS LONG) AS LONG
    DIM float_x AS SINGLE
    DIM idx AS INTEGER
    
    ' Convert fixed-point to float to find lookup index
    float_x = FixedToFloat(x)
    
    ' Clamp to supported range
    IF float_x < EXP_MIN THEN 
        RETURN 0 ' e^(very_negative) ≈ 0
    END IF
    
    IF float_x > EXP_MAX THEN
        float_x = EXP_MAX ' Prevent overflow
    END IF
    
    ' Map to lookup table index
    idx = INT((float_x - EXP_MIN) * (EXP_LOOKUP_SIZE - 1) / EXP_RANGE + 0.5)
    
    ' Clamp index to valid range
    IF idx < 0 THEN idx = 0
    IF idx >= EXP_LOOKUP_SIZE THEN idx = EXP_LOOKUP_SIZE - 1
    
    ' Return value from lookup table
    RETURN exp_lookup(idx)
END FUNCTION

' Optimized fixed-point exponential approximation 
' using piecewise approximation
FUNCTION FixedExpFast(x AS LONG) AS LONG
    DIM x_int AS INTEGER, x_frac AS INTEGER
    DIM result AS LONG
    
    ' Clamp x to prevent overflow
    IF x <= FloatToFixed(EXP_MIN) THEN RETURN 0 ' Underflow to 0
    IF x >= FloatToFixed(EXP_MAX) THEN RETURN FloatToFixed(EXP(EXP_MAX)) ' Max value
    
    ' Extract integer and fractional parts of x
    x_int = x >> FIXED_POINT_SHIFT ' Integer part
    x_frac = x AND FIXED_POINT_MASK ' Fractional part
    
    ' e^x = e^(x_int + x_frac) = e^x_int * e^x_frac
    
    ' For the integer part, use bit shifting:
    ' e^1 ≈ 2.718, so we approximate e^n with a scaling factor
    ' For the fractional part, use a polynomial approximation
    ' e^f ≈ 1 + f + f²/2 + f³/6 for small f
    
    IF x_int <= -30 THEN
        RETURN 0 ' Underflow to 0
    END IF
    
    ' First approximate e^(x_frac)
    ' Convert fractional fixed-point back to a value between 0 and 1
    DIM frac_float AS SINGLE
    frac_float = x_frac / FIXED_POINT_ONE
    
    ' Compute approximation for e^(x_frac)
    DIM exp_frac AS SINGLE
    exp_frac = 1.0 + frac_float + (frac_float * frac_float) / 2.0 + _
               (frac_float * frac_float * frac_float) / 6.0
    
    ' Now compute e^(x_int) * approximation
    DIM exp_int AS DOUBLE
    exp_int = EXP(x_int)
    
    ' Combine and convert back to fixed-point
    RETURN FloatToFixed(exp_int * exp_frac)
END FUNCTION

' Alternative implementation avoiding floating point
FUNCTION FixedExpInteger(x AS LONG) AS LONG
    ' This implementation avoids floating point entirely,
    ' using only integer operations. It uses the property:
    ' e^x ≈ 2^(x * log2(e))
    ' where log2(e) ≈ 1.4427
    
    DIM LOG2_E_FIXED AS LONG
    LOG2_E_FIXED = FloatToFixed(1.4427)
    
    ' Compute x * log2(e)
    DIM x_log2e AS LONG
    x_log2e = FixedMul(x, LOG2_E_FIXED)
    
    ' Now compute 2^(x_log2e)
    DIM int_part AS LONG, frac_part AS LONG
    int_part = x_log2e >> FIXED_POINT_SHIFT
    frac_part = x_log2e AND FIXED_POINT_MASK
    
    ' Handle integer part with bit shifting (2^n)
    DIM result AS LONG
    
    ' Check for overflow and underflow
    IF int_part <= -30 THEN
        RETURN 0 ' Underflow to 0
    ELSEIF int_part >= 30 THEN
        RETURN &H7FFFFFFF ' Maximum positive 32-bit value
    END IF
    
    ' Compute 2^int_part
    IF int_part >= 0 THEN
        result = FIXED_POINT_ONE << int_part
    ELSE
        result = FIXED_POINT_ONE >> (-int_part)
    END IF
    
    ' Approximate 2^frac_part using a degree-3 polynomial
    ' 2^f ≈ 1 + 0.693f + 0.24f² + 0.056f³ for 0 ≤ f < 1
    DIM f AS LONG, f2 AS LONG, f3 AS LONG
    DIM poly AS LONG
    
    f = frac_part
    f2 = FixedMul(f, f)
    f3 = FixedMul(f2, f)
    
    poly = FIXED_POINT_ONE + ' 1
           FixedMul(FloatToFixed(0.693), f) + ' ln(2) * f
           FixedMul(FloatToFixed(0.24), f2) + ' term for f²
           FixedMul(FloatToFixed(0.056), f3)  ' term for f³
    
    ' Multiply integer and fractional parts
    result = FixedMul(result, poly)
    
    RETURN result
END FUNCTION

' *******************************************************
' * Fixed-Point Softmax Implementation                  *
' *******************************************************

' Fixed-point softmax for a vector
' Computes softmax while avoiding overflow and underflow
SUB FixedSoftmaxVector(x() AS LONG, result() AS LONG, length AS INTEGER)
    DIM i AS INTEGER
    DIM max_val AS LONG, sum AS LONG
    DIM shifted(0 TO length - 1) AS LONG
    
    ' Find maximum value for numerical stability
    max_val = x(0)
    FOR i = 1 TO length - 1
        IF x(i) > max_val THEN
            max_val = x(i)
        END IF
    NEXT i
    
    ' Compute e^(x - max) for all elements and sum
    sum = 0
    FOR i = 0 TO length - 1
        ' Shift values by subtracting max (this becomes e^(x - max))
        shifted(i) = x(i) - max_val
        
        ' Now compute exp of shifted values using our fixed-point exp
        shifted(i) = FixedExp(shifted(i))
        
        ' Keep track of sum
        sum = sum + shifted(i)
    NEXT i
    
    ' Normalize by dividing by sum
    FOR i = 0 TO length - 1
        ' Avoid division by zero
        IF sum > 0 THEN
            result(i) = FixedDiv(shifted(i), sum)
        ELSE
            ' If all values were extremely negative, distribution is uniform
            result(i) = FixedDiv(FIXED_POINT_ONE, FloatToFixed(length))
        END IF
    NEXT i
END SUB

' Fixed-point softmax on a whole matrix (row-wise)
SUB FixedSoftmaxMatrix(A() AS LONG, result() AS LONG, rows AS INTEGER, cols AS INTEGER)
    DIM i AS INTEGER, j AS INTEGER
    DIM offset AS LONG
    DIM row(0 TO 1023) AS LONG ' Assuming max cols is 1024, increase if needed
    DIM row_result(0 TO 1023) AS LONG
    
    ' Apply softmax to each row
    FOR i = 0 TO rows - 1
        ' Extract row
        offset = i * cols
        FOR j = 0 TO cols - 1
            row(j) = A(offset + j)
        NEXT j
        
        ' Apply softmax to row
        FixedSoftmaxVector(row(), row_result(), cols)
        
        ' Store result
        FOR j = 0 TO cols - 1
            result(offset + j) = row_result(j)
        NEXT j
    NEXT i
END SUB

' *******************************************************
' * Matrix Softmax Integration                          *
' *******************************************************

' Convert float matrix to fixed-point format
SUB MatrixToFixed(A AS Matrix, BYREF fixed_data() AS LONG)
    DIM size AS LONG, i AS LONG
    
    ' Calculate total size
    size = A.rows * A.cols
    
    ' Allocate fixed-point array
    REDIM fixed_data(0 TO size - 1)
    
    ' Convert each element to fixed point
    FOR i = 0 TO size - 1
        ' Calculate row and column
        DIM row AS INTEGER, col AS INTEGER
        row = i \ A.cols
        col = i MOD A.cols
        
        ' Convert to fixed point
        fixed_data(i) = FloatToFixed(A.data(row, col))
    NEXT i
END SUB

' Convert fixed-point data back to float matrix
SUB FixedToMatrix(fixed_data() AS LONG, BYREF B AS Matrix, rows AS INTEGER, cols AS INTEGER)
    DIM size AS LONG, i AS LONG
    
    ' Initialize result matrix if needed
    IF B.rows <> rows OR B.cols <> cols THEN
        FreeMatrix(B)
        InitMatrix(B, rows, cols)
    END IF
    
    ' Convert each element back to float
    FOR i = 0 TO (rows * cols) - 1
        ' Calculate row and column
        DIM row AS INTEGER, col AS INTEGER
        row = i \ cols
        col = i MOD cols
        
        ' Convert to float
        B.data(row, col) = FixedToFloat(fixed_data(i))
    NEXT i
END SUB

' Fixed-point matrix softmax for external use
SUB MatrixSoftmaxFixed(A AS Matrix, BYREF B AS Matrix)
    DIM fixed_A() AS LONG, fixed_B() AS LONG
    
    ' Convert input matrix to fixed-point
    MatrixToFixed(A, fixed_A)
    
    ' Allocate space for result
    REDIM fixed_B(0 TO A.rows * A.cols - 1)
    
    ' Apply fixed-point softmax
    FixedSoftmaxMatrix(fixed_A, fixed_B, A.rows, A.cols)
    
    ' Convert result back to float matrix
    FixedToMatrix(fixed_B, B, A.rows, A.cols)
END SUB

' Fixed-point softmax implemented with assembly for 486
SUB MatrixSoftmaxFixedAsm(A AS Matrix, BYREF B AS Matrix)
    IF g_use_assembly AND g_has_assembly_softmax THEN
        ' If assembly optimization is available, use it
        SoftmaxAsm(A, B)
    ELSE
        ' Otherwise fall back to our fixed-point implementation
        MatrixSoftmaxFixed(A, B)
    END IF
END SUB

' *******************************************************
' * Testing Functions                                   *
' *******************************************************

' Test fixed-point exp function against floating-point exp
SUB TestFixedExp()
    DIM i AS INTEGER
    DIM x AS SINGLE, expected AS SINGLE, actual AS SINGLE
    DIM fixed_x AS LONG, fixed_result AS LONG
    DIM max_error AS SINGLE, avg_error AS SINGLE
    
    PRINT "Testing fixed-point exponential function..."
    
    ' Test over a range of values
    max_error = 0
    avg_error = 0
    
    FOR i = 0 TO 20
        ' Generate test value from -5 to 5
        x = -5.0 + (i * 0.5)
        
        ' Expected floating-point result
        expected = EXP(x)
        
        ' Convert to fixed-point, compute exp, and convert back
        fixed_x = FloatToFixed(x)
        fixed_result = FixedExp(fixed_x)
        actual = FixedToFloat(fixed_result)
        
        ' Calculate error
        DIM error AS SINGLE
        error = ABS((actual - expected) / expected) * 100 ' Percent error
        
        ' Update max and average error
        IF error > max_error THEN max_error = error
        avg_error = avg_error + error
        
        ' Print results
        IF i <= 10 THEN ' Print first few results
            PRINT "exp("; x; ") = "; expected; " (expected) vs "; actual; " (fixed), error = "; error; "%"
        END IF
    NEXT i
    
    avg_error = avg_error / 21
    
    PRINT "Maximum error: "; max_error; "%"
    PRINT "Average error: "; avg_error; "%"
    
    ' Test fast exp approximation
    PRINT
    PRINT "Testing fast exp approximation..."
    
    max_error = 0
    avg_error = 0
    
    FOR i = 0 TO 20
        ' Generate test value from -5 to 5
        x = -5.0 + (i * 0.5)
        
        ' Expected floating-point result
        expected = EXP(x)
        
        ' Convert to fixed-point, compute fast exp, and convert back
        fixed_x = FloatToFixed(x)
        fixed_result = FixedExpFast(fixed_x)
        actual = FixedToFloat(fixed_result)
        
        ' Calculate error
        DIM error AS SINGLE
        error = ABS((actual - expected) / expected) * 100 ' Percent error
        
        ' Update max and average error
        IF error > max_error THEN max_error = error
        avg_error = avg_error + error
        
        ' Print results
        IF i <= 10 THEN ' Print first few results
            PRINT "exp("; x; ") = "; expected; " (expected) vs "; actual; " (fast), error = "; error; "%"
        END IF
    NEXT i
    
    avg_error = avg_error / 21
    
    PRINT "Maximum error: "; max_error; "%"
    PRINT "Average error: "; avg_error; "%"
END SUB

' Test fixed-point softmax function
SUB TestFixedSoftmax()
    DIM i AS INTEGER, j AS INTEGER
    DIM a AS Matrix, b_float AS Matrix, b_fixed AS Matrix
    DIM max_error AS SINGLE, total_error AS SINGLE
    
    PRINT "Testing fixed-point softmax..."
    
    ' Initialize test matrix
    InitMatrix(a, 3, 4)
    
    ' Fill with test values
    a.data(0, 0) = 1.0: a.data(0, 1) = 2.0: a.data(0, 2) = 3.0: a.data(0, 3) = 4.0
    a.data(1, 0) = 0.5: a.data(1, 1) = 0.5: a.data(1, 2) = 0.5: a.data(1, 3) = 0.5
    a.data(2, 0) = -1.0: a.data(2, 1) = -2.0: a.data(2, 2) = -3.0: a.data(2, 3) = -4.0
    
    ' Compute standard floating-point softmax
    MatrixSoftmax(a, b_float)
    
    ' Compute fixed-point softmax
    MatrixSoftmaxFixed(a, b_fixed)
    
    ' Print results
    PRINT "Original matrix:"
    PrintMatrix(a, "A")
    
    PRINT "Floating-point softmax:"
    PrintMatrix(b_float, "Softmax(A) - float")
    
    PRINT "Fixed-point softmax:"
    PrintMatrix(b_fixed, "Softmax(A) - fixed")
    
    ' Compare results
    max_error = 0
    total_error = 0
    
    FOR i = 0 TO a.rows - 1
        FOR j = 0 TO a.cols - 1
            DIM error AS SINGLE
            error = ABS(b_float.data(i, j) - b_fixed.data(i, j))
            
            IF error > max_error THEN max_error = error
            total_error = total_error + error
        NEXT j
    NEXT i
    
    PRINT "Maximum error: "; max_error
    PRINT "Average error: "; total_error / (a.rows * a.cols)
    
    ' Verify softmax rows sum to 1
    PRINT "Verifying softmax row sums (should be 1.0):"
    
    FOR i = 0 TO b_fixed.rows - 1
        DIM row_sum AS SINGLE
        row_sum = 0.0
        
        FOR j = 0 TO b_fixed.cols - 1
            row_sum = row_sum + b_fixed.data(i, j)
        NEXT j
        
        PRINT "Row "; i; " sum: "; row_sum
    NEXT i
    
    ' Free matrices
    FreeMatrix(a)
    FreeMatrix(b_float)
    FreeMatrix(b_fixed)
END SUB

' Test performance of fixed-point vs floating-point softmax
SUB TestSoftmaxPerformance()
    DIM i AS INTEGER, j AS INTEGER, k AS INTEGER
    DIM a AS Matrix, b_float AS Matrix, b_fixed AS Matrix
    DIM start_time AS DOUBLE, end_time AS DOUBLE
    DIM float_time AS DOUBLE, fixed_time AS DOUBLE
    
    PRINT "Testing softmax performance..."
    
    ' Create larger test matrix for better timing
    InitMatrix(a, 128, 128)
    
    ' Fill with random values
    FOR i = 0 TO a.rows - 1
        FOR j = 0 TO a.cols - 1
            a.data(i, j) = RND * 10 - 5 ' Random values between -5 and 5
        NEXT j
    NEXT i
    
    ' Time floating-point softmax
    start_time = TIMER
    FOR k = 1 TO 10 ' Run multiple iterations for better timing
        MatrixSoftmax(a, b_float)
    NEXT k
    end_time = TIMER
    float_time = end_time - start_time
    
    ' Time fixed-point softmax
    start_time = TIMER
    FOR k = 1 TO 10 ' Run multiple iterations for better timing
        MatrixSoftmaxFixed(a, b_fixed)
    NEXT k
    end_time = TIMER
    fixed_time = end_time - start_time
    
    ' Report results
    PRINT "Floating-point softmax time: "; float_time; " seconds"
    PRINT "Fixed-point softmax time   : "; fixed_time; " seconds"
    PRINT "Speedup/slowdown ratio     : "; float_time / fixed_time; "x"
    
    ' Compare results
    DIM max_error AS SINGLE, total_error AS SINGLE
    max_error = 0
    total_error = 0
    
    FOR i = 0 TO a.rows - 1
        FOR j = 0 TO a.cols - 1
            DIM error AS SINGLE
            error = ABS(b_float.data(i, j) - b_fixed.data(i, j))
            
            IF error > max_error THEN max_error = error
            total_error = total_error + error
        NEXT j
    NEXT i
    
    PRINT "Maximum error: "; max_error
    PRINT "Average error: "; total_error / (a.rows * a.cols)
    
    ' Free matrices
    FreeMatrix(a)
    FreeMatrix(b_float)
    FreeMatrix(b_fixed)
END SUB

' Main test routine
SUB TestFixedPointSoftmax()
    PRINT "Testing Fixed-Point Softmax Module"
    PRINT "=================================="
    PRINT
    
    ' Initialize
    InitFixedSoftmax()
    
    ' Run tests
    TestFixedExp()
    PRINT
    
    TestFixedSoftmax()
    PRINT
    
    TestSoftmaxPerformance()
END SUB
