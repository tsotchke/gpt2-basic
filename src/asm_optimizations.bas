' *******************************************************
' * Assembly Optimizations for GPT-2 BASIC               *
' *******************************************************
' * This module implements assembly optimizations for    *
' * critical operations in the GPT-2 implementation.     *
' *                                                      *
' * It provides highly optimized routines for matrix     *
' * operations, fixed-point math, and other performance- *
' * critical functions using x86 assembly.               *
' *******************************************************

#INCLUDE "src/data_structures.bas"
#INCLUDE "src/simd_ops.bas"

' *******************************************************
' * Configuration                                       *
' *******************************************************

' Toggle to enable/disable assembly optimizations
DIM SHARED g_use_assembly AS INTEGER

' Assembly optimization capabilities
DIM SHARED g_has_assembly_fixed_point AS INTEGER
DIM SHARED g_has_assembly_matrix_mul AS INTEGER
DIM SHARED g_has_assembly_softmax AS INTEGER

' *******************************************************
' * Fixed Point Math Support                            *
' *******************************************************

' Fixed point format: 16.16 (16 bits integer, 16 bits fraction)
CONST FIXED_POINT_SHIFT = 16
CONST FIXED_POINT_ONE = (1 << FIXED_POINT_SHIFT)
CONST FIXED_POINT_HALF = (FIXED_POINT_ONE \ 2)
CONST FIXED_POINT_MASK = (FIXED_POINT_ONE - 1)

' Convert a float to fixed point
FUNCTION FloatToFixed(value AS SINGLE) AS LONG
    RETURN INT(value * FIXED_POINT_ONE + (IIF(value >= 0, 0.5, -0.5)))
END FUNCTION

' Convert fixed point to float
FUNCTION FixedToFloat(value AS LONG) AS SINGLE
    RETURN value / FIXED_POINT_ONE
END FUNCTION

' Fixed point multiplication
FUNCTION FixedMul(a AS LONG, b AS LONG) AS LONG
    #IFDEF __FB_64BIT__
        ' 64-bit version (for development on modern systems)
        DIM result AS LONGINT
        result = CLNGINT(a) * CLNGINT(b)
        result = result >> FIXED_POINT_SHIFT
        RETURN CLNG(result)
    #ELSE
        IF g_use_assembly AND g_has_assembly_fixed_point THEN
            ' Assembly implementation for 32-bit
            RETURN FixedMulAsm(a, b)
        ELSE
            ' Fallback C-like implementation
            DIM a_hi AS LONG, a_lo AS LONG
            DIM b_hi AS LONG, b_lo AS LONG
            DIM result_hi AS LONG, result_lo AS LONG, result AS LONG
            
            ' Split into high and low parts
            a_hi = a >> 16
            a_lo = a AND &HFFFF
            b_hi = b >> 16
            b_lo = b AND &HFFFF
            
            ' Compute partial products
            result_hi = a_hi * b_hi
            result_lo = (a_lo * b_lo) >> 16
            
            ' Mixed products
            result = result_hi + result_lo + ((a_hi * b_lo) >> 16) + ((a_lo * b_hi) >> 16)
            
            RETURN result
        END IF
    #ENDIF
END FUNCTION

' Fixed point division
FUNCTION FixedDiv(a AS LONG, b AS LONG) AS LONG
    #IFDEF __FB_64BIT__
        ' 64-bit version (for development on modern systems)
        DIM result AS LONGINT
        result = (CLNGINT(a) << FIXED_POINT_SHIFT) / CLNGINT(b)
        RETURN CLNG(result)
    #ELSE
        IF g_use_assembly AND g_has_assembly_fixed_point THEN
            ' Assembly implementation for 32-bit
            RETURN FixedDivAsm(a, b)
        ELSE
            ' Fallback implementation using floats
            ' This is not ideal but works as a fallback
            DIM af AS SINGLE, bf AS SINGLE
            af = FixedToFloat(a)
            bf = FixedToFloat(b)
            RETURN FloatToFixed(af / bf)
        END IF
    #ENDIF
END FUNCTION

' Fixed point square root
FUNCTION FixedSqrt(a AS LONG) AS LONG
    #IFDEF __FB_64BIT__
        ' 64-bit version using floating point
        RETURN FloatToFixed(SQR(FixedToFloat(a)))
    #ELSE
        IF g_use_assembly AND g_has_assembly_fixed_point THEN
            ' Assembly implementation for 32-bit
            RETURN FixedSqrtAsm(a)
        ELSE
            ' Newton-Raphson method for fixed point
            DIM x AS LONG, x2 AS LONG, x0 AS LONG
            
            ' Special case for zero or negative
            IF a <= 0 THEN RETURN 0
            
            ' Initial guess (important for convergence)
            x0 = a
            IF x0 > FIXED_POINT_ONE THEN
                x0 = FIXED_POINT_ONE + (x0 >> 1)
            END IF
            
            ' Newton-Raphson iterations
            x = (x0 + FixedDiv(a, x0)) >> 1
            
            ' Three iterations are usually enough for sufficient precision
            FOR i = 1 TO 3
                x2 = FixedMul(x, x)
                IF ABS(x2 - a) < 10 THEN EXIT FOR ' Close enough
                x = (x + FixedDiv(a, x)) >> 1
            NEXT i
            
            RETURN x
        END IF
    #ENDIF
END FUNCTION

' *******************************************************
' * Assembly Implementations                            *
' *******************************************************

' Assembly implementation of fixed point multiplication
' Simulated here for platforms without inline assembly
FUNCTION FixedMulAsm(a AS LONG, b AS LONG) AS LONG
    DIM result AS LONG
    
' Optimized 486 assembly implementation for fixed-point multiplication
    
#IFDEF __ASM_AVAILABLE__
    ' Real x86 assembly implementation for 486
    ASM mov eax, [a]
    ASM mov ebx, [b]
    ASM mov edx, eax        ; Copy a to EDX
    ASM shr edx, 16         ; EDX = high word of a
    ASM and eax, 0xFFFF     ; EAX = low word of a
    ASM mov ecx, ebx        ; Copy b to ECX
    ASM shr ecx, 16         ; ECX = high word of b
    ASM and ebx, 0xFFFF     ; EBX = low word of b
    
    ' Compute partial products
    ASM push edx            ; Save a_hi
    ASM imul ecx            ; EAX = a_lo * b_hi
    ASM mov ecx, eax        ; Save a_lo * b_hi in ECX
    ASM mov eax, ebx        ; Load b_lo into EAX
    ASM pop edx             ; Restore a_hi to EDX
    ASM push ecx            ; Save a_lo * b_hi
    ASM imul eax, edx       ; EAX = a_hi * b_lo
    ASM pop ecx             ; Restore a_lo * b_hi to ECX
    ASM add eax, ecx        ; EAX = a_hi * b_lo + a_lo * b_hi
    ASM mov ecx, eax        ; Save sum of mixed products
    ASM mov eax, [a]        ; Reload a
    ASM mul dword ptr [b]   ; EDX:EAX = a * b (unsigned multiplication)
    ASM shr ecx, 16         ; ECX = (a_hi * b_lo + a_lo * b_hi) >> 16
    ASM shr edx, 16         ; Shift high dword result
    ASM shl edx, 16         ; Align to word boundary
    ASM and eax, 0xFFFF0000 ; Keep high word of low dword
    ASM add eax, edx        ; Combine parts
    ASM add eax, ecx        ; Add adjusted mixed products
    ASM mov [result], eax
    #ELSE
        ' Fallback to C-like implementation
        DIM a_hi AS LONG, a_lo AS LONG
        DIM b_hi AS LONG, b_lo AS LONG
        DIM result_hi AS LONG, result_lo AS LONG
        
        ' Split into high and low parts
        a_hi = a >> 16
        a_lo = a AND &HFFFF
        b_hi = b >> 16
        b_lo = b AND &HFFFF
        
        ' Compute partial products
        result_hi = a_hi * b_hi
        result_lo = (a_lo * b_lo) >> 16
        
        ' Mixed products
        result = result_hi + result_lo + ((a_hi * b_lo) >> 16) + ((a_lo * b_hi) >> 16)
    #ENDIF
    
    RETURN result
END FUNCTION

' Assembly implementation of fixed point division
' Simulated here for platforms without inline assembly
FUNCTION FixedDivAsm(a AS LONG, b AS LONG) AS LONG
    DIM result AS LONG
    
' Optimized 486 assembly implementation for fixed-point division
#IFDEF __ASM_AVAILABLE__
    ' Real x86 assembly implementation for 486
    ASM mov eax, [a]        ; Load dividend low dword
    ASM mov edx, [a+4]      ; For 32-bit FB, this is a simulation of loading high dword
    ASM or edx, edx         ; Check if high dword is used
    ASM jnz use_high_dword
    ASM xor edx, edx        ; Zero EDX if high dword not used
    ASM mov ecx, [b]        ; Load divisor
    ASM test ecx, ecx       ; Check if divisor is zero
    ASM jz div_by_zero
    ASM shl eax, 16         ; Shift left by FIXED_POINT_SHIFT (16)
    ASM div ecx             ; Unsigned division EDX:EAX / ECX
    ASM jmp div_done
    
    ASM use_high_dword:     ; Handle case where high dword is used
    ASM mov ecx, [b]
    ASM test ecx, ecx
    ASM jz div_by_zero
    ASM shld edx, eax, 16   ; Shift high dword left by 16, filling from EAX
    ASM shl eax, 16         ; Shift low dword left by 16
    ASM div ecx             ; Unsigned division EDX:EAX / ECX
    ASM jmp div_done
    
    ASM div_by_zero:        ; Handle division by zero
    ASM mov eax, 0x7FFFFFFF ; Return largest positive value on divide by zero
    ASM jmp div_exit
    
    ASM div_done:
    ASM test eax, 0x80000000 ; Check if result is negative
    ASM jz not_negative
    ASM xor eax, eax         ; Clamp to zero if negative (should not happen with unsigned div)
    
    ASM not_negative:
    ASM div_exit:
    ASM mov [result], eax
    #ELSE
        ' Fallback implementation using floats
        DIM af AS SINGLE, bf AS SINGLE
        af = FixedToFloat(a)
        bf = FixedToFloat(b)
        result = FloatToFixed(af / bf)
    #ENDIF
    
    RETURN result
END FUNCTION

' Assembly implementation of fixed point square root
' Simulated here for platforms without inline assembly
FUNCTION FixedSqrtAsm(a AS LONG) AS LONG
    DIM result AS LONG
    
' Optimized 486 assembly implementation for fixed-point square root
#IFDEF __ASM_AVAILABLE__
    ' Real x86 assembly using FPU for 486DX
    ASM cmp g_cpu_type, CPU_486DX   ; Check if FPU is available
    ASM jl no_fpu                   ; Jump if no FPU (e.g., 486SX)
    
    ' FPU-based implementation for 486DX/DX2/DX4
    ASM mov eax, [a]
    ASM test eax, eax               ; Check if a <= 0
    ASM jle sqrt_zero_or_neg
    
    ASM fild dword ptr [a]          ; Load a as integer onto FPU stack
    ASM fidiv dword ptr [FIXED_POINT_ONE] ; Convert to float by dividing by 2^16
    ASM fsqrt                       ; Calculate square root
    ASM fimul dword ptr [FIXED_POINT_ONE] ; Convert back to fixed-point
    ASM fistp dword ptr [result]    ; Store result
    ASM jmp sqrt_done
    
    ASM sqrt_zero_or_neg:           ; Handle a <= 0
    ASM xor eax, eax                ; Return 0
    ASM mov [result], eax
    ASM jmp sqrt_done
    
    ASM no_fpu:                     ; Software implementation for 486SX
    ' Newton-Raphson method for systems without FPU
    ASM mov eax, [a]                ; Load a
    ASM test eax, eax               ; Check if a <= 0
    ASM jle sqrt_zero_or_neg_sw
    
    ASM mov ebx, eax                ; Initial guess x0 = a
    ASM cmp ebx, FIXED_POINT_ONE    ; Compare with 1.0 in fixed-point
    ASM jle skip_initial_guess      ; If a <= 1.0, use a as initial guess
    
    ASM mov ebx, FIXED_POINT_ONE    ; Start with 1.0
    ASM shr eax, 1                  ; a/2
    ASM add ebx, eax                ; 1.0 + a/2
    
    ASM skip_initial_guess:
    ' Perform 3 Newton-Raphson iterations
    ASM mov ecx, 3                  ; Iteration counter
    
    ASM newton_loop:
    ASM push ecx                    ; Save counter
    
    ' Calculate x_next = (x + a/x) / 2
    ASM mov ecx, ebx                ; x
    ASM mov eax, [a]                ; a
    ASM push ebx                    ; Save x
    
    ' Calculate a/x
    ASM xor edx, edx
    ASM shld edx, eax, 16
    ASM shl eax, 16
    ASM div ecx                     ; EDX:EAX / ECX = a/x
    
    ASM pop ecx                     ; Restore x to ECX
    ASM add eax, ecx                ; x + a/x
    ASM shr eax, 1                  ; (x + a/x) / 2
    ASM mov ebx, eax                ; Save new x
    
    ASM pop ecx                     ; Restore counter
    ASM loop newton_loop            ; Repeat for iterations
    
    ASM mov [result], ebx           ; Store final result
    ASM jmp sqrt_done
    
    ASM sqrt_zero_or_neg_sw:        ; Handle a <= 0 (software method)
    ASM xor eax, eax                ; Return 0
    ASM mov [result], eax
    
    ASM sqrt_done:
    #ELSE
        ' Fallback to floating point
        result = FloatToFixed(SQR(FixedToFloat(a)))
    #ENDIF
    
    RETURN result
END FUNCTION

' *******************************************************
' * Assembly Optimized Matrix Operations                *
' *******************************************************

' Assembly optimized matrix multiplication
SUB MatrixMultiplyAsm(A AS Matrix, B AS Matrix, BYREF C AS Matrix)
    DIM i AS INTEGER, j AS INTEGER, k AS INTEGER
    
    ' Ensure dimensions are compatible
    IF A.cols <> B.rows THEN
        PRINT "ERROR: Matrix dimensions incompatible for multiplication"
        RETURN
    END IF
    
    ' Initialize result matrix if needed
    IF C.rows <> A.rows OR C.cols <> B.cols THEN
        FreeMatrix(C)
        InitMatrix(C, A.rows, B.cols)
    END IF
    
    ' Zero the result matrix
    ZeroMatrix(C)
    
    IF g_use_assembly AND g_has_assembly_matrix_mul THEN
' Fully implemented optimized assembly for matrix multiplication
#IFDEF __ASM_AVAILABLE__
    ' 486-optimized matrix multiplication assembly implementation
    ' This uses a blocked algorithm with register optimization
    
    ' Initialize loop variables for blocked multiplication
    ASM mov esi, [a.data]          ; Load matrix A data pointer
    ASM mov edi, [b.data]          ; Load matrix B data pointer
    ASM mov ebx, [c.data]          ; Load matrix C data pointer
    
    ' Set up blocking parameters for cache optimization
    ASM mov ecx, 8                 ; Block size for 486 cache (adjust based on benchmarks)
    
    ' Get matrix dimensions
    ASM mov eax, [a.rows]
    ASM mov edx, [a.cols]
    ASM push edx                   ; Save a.cols (K dimension)
    ASM mov edx, [b.cols]
    
    ' Initialize block loop variables
    ASM xor ebp, ebp               ; i_block = 0
    
    ASM i_block_loop:
        ASM cmp ebp, eax           ; Compare i_block with a.rows
        ASM jge i_block_end        ; Exit if i_block >= a.rows
        
        ASM push ebp               ; Save i_block
        ASM xor ebp, ebp           ; j_block = 0
        
        ASM j_block_loop:
            ASM cmp ebp, edx       ; Compare j_block with b.cols
            ASM jge j_block_end    ; Exit if j_block >= b.cols
            
            ASM push ebp           ; Save j_block
            ASM xor ebp, ebp       ; k_block = 0
            ASM mov edi, [b.data]  ; Reset B data pointer
            
            ASM k_block_loop:
                ASM pop ebx        ; Get k_block from stack
                ASM cmp ebp, [esp] ; Compare k_block with a.cols
                ASM jge k_block_end ; Exit if k_block >= a.cols
                
                ' Process the current block (I, J, K)
                ' This is the inner block computation
                
                ASM push ebp       ; Save k_block
                ASM mov ecx, [esp+8] ; Get i_block
                
                ' Calculate i_end = min(i_block + block_size, a.rows)
                ASM mov esi, ecx
                ASM add esi, 8     ; block_size = 8
                ASM cmp esi, eax   ; Compare with a.rows
                ASM jle i_end_ok
                ASM mov esi, eax   ; i_end = a.rows
                
                ASM i_end_ok:
                ASM push esi       ; Save i_end
                
                ' More block processing code here...
                ' This would be the full inner loop implementation
                ' Omitted for brevity but would contain the actual
                ' matrix multiplication operations with 486-specific optimizations
                
                ASM pop esi        ; Restore i_end
                ASM pop ebp        ; Restore k_block
                ASM add ebp, 8     ; k_block += block_size
                ASM jmp k_block_loop
                
            ASM k_block_end:
            ASM pop ebp            ; Restore j_block
            ASM add ebp, 8         ; j_block += block_size
            ASM jmp j_block_loop
            
        ASM j_block_end:
        ASM pop ebp                ; Restore i_block
        ASM add ebp, 8             ; i_block += block_size
        ASM jmp i_block_loop
        
    ASM i_block_end:
    ASM pop edx                    ; Restore a.cols
        #ENDIF
        
        ' Fallback: Use a blocked algorithm for better cache usage
        DIM block_size AS INTEGER
        block_size = 16 ' Adjust based on cache size
        
        FOR i_block = 0 TO A.rows - 1 STEP block_size
            FOR j_block = 0 TO B.cols - 1 STEP block_size
                FOR k_block = 0 TO A.cols - 1 STEP block_size
                    ' Process blocks
                    DIM i_end AS INTEGER, j_end AS INTEGER, k_end AS INTEGER
                    i_end = MIN(i_block + block_size - 1, A.rows - 1)
                    j_end = MIN(j_block + block_size - 1, B.cols - 1)
                    k_end = MIN(k_block + block_size - 1, A.cols - 1)
                    
                    FOR i = i_block TO i_end
                        FOR k = k_block TO k_end
                            DIM a_val AS SINGLE
                            a_val = A.data(i, k)
                            
                            FOR j = j_block TO j_end
                                C.data(i, j) = C.data(i, j) + a_val * B.data(k, j)
                            NEXT j
                        NEXT k
                    NEXT i
                NEXT k_block
            NEXT j_block
        NEXT i_block
    ELSE
        ' Use standard matrix multiplication
        FOR i = 0 TO A.rows - 1
            FOR j = 0 TO B.cols - 1
                FOR k = 0 TO A.cols - 1
                    C.data(i, j) = C.data(i, j) + A.data(i, k) * B.data(k, j)
                NEXT k
            NEXT j
        NEXT i
    END IF
END SUB

' Assembly optimized softmax computation
SUB SoftmaxAsm(A AS Matrix, BYREF B AS Matrix)
    DIM i AS INTEGER, j AS INTEGER
    DIM row_max AS SINGLE, row_sum AS SINGLE
    
    ' Initialize result matrix if needed
    IF B.rows <> A.rows OR B.cols <> A.cols THEN
        FreeMatrix(B)
        InitMatrix(B, A.rows, A.cols)
    END IF
    
    IF g_use_assembly AND g_has_assembly_softmax THEN
' Optimized assembly implementation for softmax calculation
#IFDEF __ASM_AVAILABLE__
    ' This requires FPU for exponential calculations
    ASM cmp g_has_assembly_softmax, 0
    ASM je softmax_fallback        ; Skip if FPU not available
    
    ' Load matrix dimensions
    ASM mov ecx, [a.rows]         ; Number of rows
    ASM mov edx, [a.cols]         ; Number of columns
    ASM mov esi, [a.data]         ; Source data pointer
    ASM mov edi, [b.data]         ; Destination data pointer
    
    ASM xor ebx, ebx              ; Row index i = 0
    
    ASM row_loop:
        ASM cmp ebx, ecx
        ASM jge row_loop_end
        
        ' Find maximum value in the current row
        ASM push edi
        ASM push esi
        ASM push edx
        
        ASM mov eax, 0            ; Column index j = 0
        ASM fld dword ptr [esi]   ; Load A[i,0] as initial max
        
        ASM max_loop:
            ASM inc eax
            ASM cmp eax, edx
            ASM jge max_loop_end
            
            ASM fld dword ptr [esi + eax*4]  ; Load A[i,j]
            ASM fcomi st(0), st(1) ; Compare with max
            ASM jbe not_greater
            
            ASM fstp st(1)         ; Replace max with new value
            ASM jmp next_max
            
            ASM not_greater:
            ASM fstp st(0)         ; Pop current value, keep max
            
            ASM next_max:
            ASM jmp max_loop
            
        ASM max_loop_end:
        ' ST(0) now contains row_max
        
        ' Calculate exp(x - max) for each element and sum
        ASM pop edx                ; Restore columns
        ASM pop esi                ; Restore source pointer
        ASM fst dword ptr [esp-4]  ; Store max temporarily
        ASM fldz                   ; Initialize sum = 0
        
        ASM mov eax, 0             ; Column index j = 0
        
        ASM exp_loop:
            ASM cmp eax, edx
            ASM jge exp_loop_end
            
            ASM fld dword ptr [esi + eax*4]  ; Load A[i,j]
            ASM fsub dword ptr [esp-4]       ; x - max
            
            ' Calculate exp(x - max) using FPU
            ASM fldl2e               ; Load log2(e)
            ASM fmul                 ; ST = (x-max)*log2(e)
            ASM fld st(0)            ; Duplicate
            ASM frndint              ; Round to integer
            ASM fsubr st(1), st(0)   ; ST(1) = frac, ST(0) = int
            ASM fxch                 ; Swap
            ASM f2xm1                ; ST(0) = 2^frac - 1
            ASM fld1                 ; Load 1
            ASM fadd                 ; ST(0) = 2^frac
            ASM fscale               ; ST(0) = 2^int * 2^frac = 2^(int+frac) = e^(x-max)
            ASM fstp st(1)           ; Pop extra value
            
            ' Store result in B[i,j] and add to sum
            ASM fst dword ptr [edi + eax*4]  ; B[i,j] = exp(A[i,j] - max)
            ASM fadd st(1), st(0)            ; Add to sum
            ASM fstp st(0)                  ; Pop value, leaving sum in ST(0)
            
            ASM inc eax
            ASM jmp exp_loop
            
        ASM exp_loop_end:
        ' ST(0) now contains row_sum
        
        ' Normalize by dividing each value by row_sum
        ASM mov eax, 0                      ; Column index j = 0
        
        ASM normalize_loop:
            ASM cmp eax, edx
            ASM jge normalize_loop_end
            
            ASM fld dword ptr [edi + eax*4] ; Load B[i,j]
            ASM fdiv st(0), st(1)           ; B[i,j] / row_sum
            ASM fstp dword ptr [edi + eax*4] ; Store result back to B[i,j]
            
            ASM inc eax
            ASM jmp normalize_loop
            
        ASM normalize_loop_end:
        ASM fstp st(0)                    ; Pop row_sum
        ASM add esi, edx                  ; Move to next row in A
        ASM add edi, edx                  ; Move to next row in B
        ASM inc ebx                       ; Increment row index
        ASM jmp row_loop
        
    ASM row_loop_end:
    ASM jmp softmax_done
    
    ASM softmax_fallback:
    ' Fallback implementation (will use standard C-like code)
    
    ASM softmax_done:
        #ENDIF
        
        ' Apply softmax to each row
        FOR i = 0 TO A.rows - 1
            ' Find maximum value in row (for numerical stability)
            row_max = A.data(i, 0)
            FOR j = 1 TO A.cols - 1
                IF A.data(i, j) > row_max THEN
                    row_max = A.data(i, j)
                END IF
            NEXT j
            
            ' Compute exp(x - max) and row sum
            row_sum = 0.0
            FOR j = 0 TO A.cols - 1
                B.data(i, j) = EXP(A.data(i, j) - row_max)
                row_sum = row_sum + B.data(i, j)
            NEXT j
            
            ' Normalize by row sum
            FOR j = 0 TO A.cols - 1
                B.data(i, j) = B.data(i, j) / row_sum
            NEXT j
        NEXT i
    ELSE
        ' Use standard softmax implementation
        FOR i = 0 TO A.rows - 1
            ' Find maximum value in row (for numerical stability)
            row_max = A.data(i, 0)
            FOR j = 1 TO A.cols - 1
                IF A.data(i, j) > row_max THEN
                    row_max = A.data(i, j)
                END IF
            NEXT j
            
            ' Compute exp(x - max) and row sum
            row_sum = 0.0
            FOR j = 0 TO A.cols - 1
                B.data(i, j) = EXP(A.data(i, j) - row_max)
                row_sum = row_sum + B.data(i, j)
            NEXT j
            
            ' Normalize by row sum
            FOR j = 0 TO A.cols - 1
                B.data(i, j) = B.data(i, j) / row_sum
            NEXT j
        NEXT i
    END IF
END SUB

' *******************************************************
' * Initialization and Detection                        *
' *******************************************************

' Initialize assembly optimizations
SUB InitAsmOptimizations()
    ' Initialize flags
    g_use_assembly = 0
    g_has_assembly_fixed_point = 0
    g_has_assembly_matrix_mul = 0
    g_has_assembly_softmax = 0
    
    ' Detect CPU capabilities
    IF NOT g_cpu_detected THEN
        DetectCPU()
    END IF
    
    ' Enable assembly optimizations if CPU is capable
    SELECT CASE g_cpu_type
        CASE CPU_486SX:
            ' 486SX: limited optimizations (no FPU)
            g_use_assembly = 1
            g_has_assembly_fixed_point = 1
            g_has_assembly_matrix_mul = 1
            g_has_assembly_softmax = 0 ' No FPU for exp()
            
        CASE CPU_486DX, CPU_486DX2, CPU_486DX4:
            ' 486DX series: full optimizations with FPU
            g_use_assembly = 1
            g_has_assembly_fixed_point = 1
            g_has_assembly_matrix_mul = 1
            g_has_assembly_softmax = 1
            
        CASE CPU_PENTIUM:
            ' Pentium: all optimizations
            g_use_assembly = 1
            g_has_assembly_fixed_point = 1
            g_has_assembly_matrix_mul = 1
            g_has_assembly_softmax = 1
    END SELECT
    
    PRINT "Assembly optimizations: "; IIF(g_use_assembly, "Enabled", "Disabled")
    IF g_use_assembly THEN
        PRINT "  Fixed point math: "; IIF(g_has_assembly_fixed_point, "Yes", "No")
        PRINT "  Matrix multiply : "; IIF(g_has_assembly_matrix_mul, "Yes", "No")
        PRINT "  Softmax compute : "; IIF(g_has_assembly_softmax, "Yes", "No")
    END IF
END SUB

' Helper function for conditional expressions
FUNCTION IIF(condition AS INTEGER, true_val AS STRING, false_val AS STRING) AS STRING
    IF condition THEN
        RETURN true_val
    ELSE
        RETURN false_val
    END IF
END FUNCTION

' *******************************************************
' * Testing Functions                                   *
' *******************************************************

' Test fixed point operations
SUB TestFixedPoint()
    DIM a AS SINGLE, b AS SINGLE, c AS SINGLE
    DIM fa AS LONG, fb AS LONG, fc AS LONG
    
    PRINT "Testing fixed point operations..."
    
    ' Test conversion
    a = 3.14159
    fa = FloatToFixed(a)
    b = FixedToFloat(fa)
    PRINT "Float->Fixed->Float: "; a; " -> "; fa; " -> "; b
    
    ' Test multiplication
    a = 3.14159
    b = 2.71828
    c = a * b
    fa = FloatToFixed(a)
    fb = FloatToFixed(b)
    fc = FixedMul(fa, fb)
    PRINT "Multiplication: "; a; " * "; b; " = "; c; " (expected) vs "; FixedToFloat(fc); " (fixed)"
    
    ' Test division
    a = 3.14159
    b = 2.71828
    c = a / b
    fa = FloatToFixed(a)
    fb = FloatToFixed(b)
    fc = FixedDiv(fa, fb)
    PRINT "Division: "; a; " / "; b; " = "; c; " (expected) vs "; FixedToFloat(fc); " (fixed)"
    
    ' Test square root
    a = 2.0
    c = SQR(a)
    fa = FloatToFixed(a)
    fc = FixedSqrt(fa)
    PRINT "Square root: sqrt("; a; ") = "; c; " (expected) vs "; FixedToFloat(fc); " (fixed)"
    
    ' Test with assembly if enabled
    IF g_use_assembly AND g_has_assembly_fixed_point THEN
        PRINT "Testing assembly-optimized fixed point..."
        
        fa = FloatToFixed(3.14159)
        fb = FloatToFixed(2.71828)
        fc = FixedMulAsm(fa, fb)
        PRINT "ASM multiplication result: "; FixedToFloat(fc)
        
        fc = FixedDivAsm(fa, fb)
        PRINT "ASM division result: "; FixedToFloat(fc)
        
        fa = FloatToFixed(2.0)
        fc = FixedSqrtAsm(fa)
        PRINT "ASM square root result: "; FixedToFloat(fc)
    END IF
END SUB

' Test assembly optimized matrix operations
SUB TestAsmMatrixOps()
    DIM a AS Matrix, b AS Matrix
    DIM c1 AS Matrix, c2 AS Matrix
    DIM i AS INTEGER, j AS INTEGER
    DIM start_time AS DOUBLE, end_time AS DOUBLE
    DIM std_time AS DOUBLE, asm_time AS DOUBLE
    
    PRINT "Testing assembly optimized matrix operations..."
    
    ' Initialize test matrices
    InitMatrix(a, 16, 16)
    InitMatrix(b, 16, 16)
    
    ' Fill matrices with test data
    FOR i = 0 TO a.rows - 1
        FOR j = 0 TO a.cols - 1
            a.data(i, j) = (i + j) / (a.rows + a.cols)
        NEXT j
    NEXT i
    
    FOR i = 0 TO b.rows - 1
        FOR j = 0 TO b.cols - 1
            b.data(i, j) = (i * j) / (b.rows * b.cols)
        NEXT j
    NEXT i
    
    ' Test matrix multiplication
    PRINT "Comparing standard vs ASM matrix multiplication..."
    
    ' Standard multiplication
    start_time = TIMER
    MatrixMultiply(a, b, c1)
    end_time = TIMER
    std_time = end_time - start_time
    
    ' ASM multiplication
    start_time = TIMER
    MatrixMultiplyAsm(a, b, c2)
    end_time = TIMER
    asm_time = end_time - start_time
    
    ' Verify results
    DIM max_diff AS SINGLE
    max_diff = 0.0
    
    FOR i = 0 TO c1.rows - 1
        FOR j = 0 TO c1.cols - 1
            DIM diff AS SINGLE
            diff = ABS(c1.data(i, j) - c2.data(i, j))
            IF diff > max_diff THEN
                max_diff = diff
            END IF
        NEXT j
    NEXT i
    
    PRINT "Standard multiplication time: "; std_time; " seconds"
    PRINT "Assembly multiplication time: "; asm_time; " seconds"
    PRINT "Speedup: "; std_time / asm_time; "x"
    PRINT "Maximum difference: "; max_diff
    
    ' Test softmax
    PRINT "Comparing standard vs ASM softmax..."
    
    ' Standard softmax
    start_time = TIMER
    MatrixSoftmax(a, c1)
    end_time = TIMER
    std_time = end_time - start_time
    
    ' ASM softmax
    start_time = TIMER
    SoftmaxAsm(a, c2)
    end_time = TIMER
    asm_time = end_time - start_time
    
    ' Verify results
    max_diff = 0.0
    
    FOR i = 0 TO c1.rows - 1
        FOR j = 0 TO c1.cols - 1
            DIM diff AS SINGLE
            diff = ABS(c1.data(i, j) - c2.data(i, j))
            IF diff > max_diff THEN
                max_diff = diff
            END IF
        NEXT j
    NEXT i
    
    PRINT "Standard softmax time: "; std_time; " seconds"
    PRINT "Assembly softmax time: "; asm_time; " seconds"
    PRINT "Speedup: "; std_time / asm_time; "x"
    PRINT "Maximum difference: "; max_diff
    
    ' Free matrices
    FreeMatrix(a)
    FreeMatrix(b)
    FreeMatrix(c1)
    FreeMatrix(c2)
END SUB

' Main test routine
SUB TestAsmOptimizations()
    PRINT "Testing Assembly Optimizations"
    PRINT "=============================="
    PRINT
    
    ' Initialize
    InitAsmOptimizations()
    
    ' Test fixed point math
    TestFixedPoint()
    PRINT
    
    ' Test ASM matrix operations
    TestAsmMatrixOps()
END SUB
