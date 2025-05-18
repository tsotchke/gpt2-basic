' *******************************************************
' * Memory Management for GPT-2 BASIC                   *
' *******************************************************
' * This module implements memory tracking, pooling,    *
' * and parameter streaming to enable transformer       *
' * models to operate within the limited memory         *
' * constraints of 486-era hardware.                    *
' *                                                      *
' * It provides mechanisms to stream model parameters   *
' * from disk on demand, track and limit memory usage,  *
' * and efficiently pool matrix allocations to avoid    *
' * fragmentation.                                      *
' *******************************************************

#INCLUDE "src/data_structures.bas"
#INCLUDE "src/matrix_ops.bas"
#INCLUDE "src/block_sparse.bas"

' *******************************************************
' * Constants and Global Variables                      *
' *******************************************************

' Memory limits (in bytes)
CONST MAX_MEMORY_USAGE = 32 * 1024 * 1024 ' 32MB maximum (typical 486 limit)
CONST SAFETY_MARGIN = 4 * 1024 * 1024 ' 4MB safety margin

' Memory pool size thresholds for matrices
CONST SMALL_MATRIX_THRESHOLD = 1024 ' 1KB
CONST MEDIUM_MATRIX_THRESHOLD = 16384 ' 16KB
CONST LARGE_MATRIX_THRESHOLD = 262144 ' 256KB

' Global memory tracking
DIM SHARED g_current_memory_usage AS LONG
DIM SHARED g_peak_memory_usage AS LONG
DIM SHARED g_matrix_allocs AS LONG
DIM SHARED g_matrix_frees AS LONG
DIM SHARED g_cache_hits AS LONG
DIM SHARED g_cache_misses AS LONG
DIM SHARED g_disk_reads AS LONG
DIM SHARED g_disk_writes AS LONG

' Pooling system
TYPE MatrixPoolEntry
    in_use AS INTEGER
    size AS LONG
    matrix_ptr AS Matrix PTR
    next AS MatrixPoolEntry PTR
END TYPE

DIM SHARED g_small_pool_head AS MatrixPoolEntry PTR
DIM SHARED g_medium_pool_head AS MatrixPoolEntry PTR
DIM SHARED g_large_pool_head AS MatrixPoolEntry PTR
DIM SHARED g_initialized AS INTEGER

' Parameter streaming system
TYPE LayerCache
    layer_idx AS INTEGER
    is_loaded AS INTEGER
    size AS LONG
    last_used AS DOUBLE
    weights() AS Matrix
    next AS LayerCache PTR
END TYPE

DIM SHARED g_layer_cache_head AS LayerCache PTR
DIM SHARED g_layer_cache_count AS INTEGER
DIM SHARED g_max_cached_layers AS INTEGER = 3 ' Number of layers to keep in memory
DIM SHARED g_model_path AS STRING

' *******************************************************
' * Memory Manager Initialization                       *
' *******************************************************

' Initialize the memory management system
SUB InitMemoryManager()
    IF g_initialized THEN
        RETURN
    END IF
    
    ' Reset memory tracking
    g_current_memory_usage = 0
    g_peak_memory_usage = 0
    g_matrix_allocs = 0
    g_matrix_frees = 0
    g_cache_hits = 0
    g_cache_misses = 0
    g_disk_reads = 0
    g_disk_writes = 0
    
    ' Initialize pools
    g_small_pool_head = NULL
    g_medium_pool_head = NULL
    g_large_pool_head = NULL
    
    ' Initialize layer cache
    g_layer_cache_head = NULL
    g_layer_cache_count = 0
    
    g_initialized = 1
    
    PRINT "Memory Manager Initialized. Maximum memory: "; MAX_MEMORY_USAGE / (1024 * 1024); "MB"
END SUB

' Shutdown and cleanup the memory management system
SUB ShutdownMemoryManager()
    ' Free all pooled matrices
    FreeMatrixPool(g_small_pool_head)
    FreeMatrixPool(g_medium_pool_head)
    FreeMatrixPool(g_large_pool_head)
    
    ' Free layer cache
    FreeAllLayerCache()
    
    g_initialized = 0
    
    PRINT "Memory Manager Shutdown. Peak memory usage: "; g_peak_memory_usage / (1024 * 1024); "MB"
    PRINT "Matrix allocations: "; g_matrix_allocs; ", Frees: "; g_matrix_frees
    PRINT "Cache hits: "; g_cache_hits; ", Misses: "; g_cache_misses
    PRINT "Disk reads: "; g_disk_reads; ", Writes: "; g_disk_writes
END SUB

' *******************************************************
' * Memory Tracking Functions                           *
' *******************************************************

' Track memory allocation
SUB TrackAllocation(size AS LONG)
    g_current_memory_usage = g_current_memory_usage + size
    
    ' Update peak usage
    IF g_current_memory_usage > g_peak_memory_usage THEN
        g_peak_memory_usage = g_current_memory_usage
    END IF
END SUB

' Track memory deallocation
SUB TrackDeallocation(size AS LONG)
    g_current_memory_usage = g_current_memory_usage - size
    
    ' Sanity check
    IF g_current_memory_usage < 0 THEN
        PRINT "WARNING: Memory tracking error - negative memory usage"
        g_current_memory_usage = 0
    END IF
END SUB

' Check if allocating additional memory would exceed limits
FUNCTION CanAllocate(size AS LONG) AS INTEGER
    RETURN (g_current_memory_usage + size) <= (MAX_MEMORY_USAGE - SAFETY_MARGIN)
END FUNCTION

' Return the amount of available memory
FUNCTION GetAvailableMemory() AS LONG
    RETURN MAX_MEMORY_USAGE - SAFETY_MARGIN - g_current_memory_usage
END FUNCTION

' Print current memory usage statistics
SUB PrintMemoryStats()
    PRINT "Memory Usage Statistics:"
    PRINT "------------------------"
    PRINT "Current usage: "; g_current_memory_usage / 1024; "KB ("; _
          FORMAT(g_current_memory_usage * 100.0 / MAX_MEMORY_USAGE, "0.00"); "%)"
    PRINT "Peak usage   : "; g_peak_memory_usage / 1024; "KB ("; _
          FORMAT(g_peak_memory_usage * 100.0 / MAX_MEMORY_USAGE, "0.00"); "%)"
    PRINT "Available    : "; GetAvailableMemory() / 1024; "KB"
    PRINT "Matrix allocs: "; g_matrix_allocs; ", Frees: "; g_matrix_frees
    PRINT "Cache hits   : "; g_cache_hits; ", Misses: "; g_cache_misses
    PRINT "Layer cache  : "; g_layer_cache_count; " / "; g_max_cached_layers; " layers"
END SUB

' *******************************************************
' * Matrix Memory Pooling                               *
' *******************************************************

' Create a new matrix pool entry
FUNCTION CreatePoolEntry(rows AS INTEGER, cols AS INTEGER) AS MatrixPoolEntry PTR
    DIM entry AS MatrixPoolEntry PTR
    entry = NEW MatrixPoolEntry
    
    entry->in_use = 1
    entry->size = rows * cols * 4 ' 4 bytes per float
    entry->matrix_ptr = NEW Matrix
    entry->next = NULL
    
    ' Initialize the actual matrix
    InitMatrix(*(entry->matrix_ptr), rows, cols)
    
    RETURN entry
END FUNCTION

' Get a pooled matrix of the specified size
' Returns a pointer to a Matrix struct
FUNCTION GetPooledMatrix(rows AS INTEGER, cols AS INTEGER) AS Matrix PTR
    DIM size AS LONG, entry AS MatrixPoolEntry PTR, pool_head AS MatrixPoolEntry PTR PTR
    
    size = rows * cols * 4 ' 4 bytes per float
    
    ' Determine which pool to use based on size
    IF size <= SMALL_MATRIX_THRESHOLD THEN
        pool_head = @g_small_pool_head
    ELSEIF size <= MEDIUM_MATRIX_THRESHOLD THEN
        pool_head = @g_medium_pool_head
    ELSEIF size <= LARGE_MATRIX_THRESHOLD THEN
        pool_head = @g_large_pool_head
    ELSE
        ' For very large matrices, always create a new one (not pooled)
        entry = CreatePoolEntry(rows, cols)
        TrackAllocation(size + SIZEOF(MatrixPoolEntry) + SIZEOF(Matrix))
        g_matrix_allocs = g_matrix_allocs + 1
        RETURN entry->matrix_ptr
    END IF
    
    ' Try to find an unused entry in the pool that's large enough
    entry = *pool_head
    WHILE entry <> NULL
        ' If entry is not in use and big enough
        IF entry->in_use = 0 AND entry->matrix_ptr->rows >= rows AND entry->matrix_ptr->cols >= cols THEN
            ' We found a suitable entry
            entry->in_use = 1
            g_cache_hits = g_cache_hits + 1
            
            ' If exact size, return as-is
            IF entry->matrix_ptr->rows = rows AND entry->matrix_ptr->cols = cols THEN
                RETURN entry->matrix_ptr
            END IF
            
            ' Otherwise resize the matrix (but don't reallocate memory)
            entry->matrix_ptr->rows = rows
            entry->matrix_ptr->cols = cols
            
            RETURN entry->matrix_ptr
        END IF
        
        entry = entry->next
    WEND
    
    ' No suitable entry found, create a new one
    g_cache_misses = g_cache_misses + 1
    entry = CreatePoolEntry(rows, cols)
    
    ' Add to the pool
    entry->next = *pool_head
    *pool_head = entry
    
    ' Track memory usage
    TrackAllocation(size + SIZEOF(MatrixPoolEntry) + SIZEOF(Matrix))
    g_matrix_allocs = g_matrix_allocs + 1
    
    RETURN entry->matrix_ptr
END FUNCTION

' Return a matrix to the pool
SUB ReturnMatrixToPool(matrix_ptr AS Matrix PTR)
    DIM entry AS MatrixPoolEntry PTR, prev AS MatrixPoolEntry PTR, pool_head AS MatrixPoolEntry PTR PTR
    DIM size AS LONG
    
    size = matrix_ptr->rows * matrix_ptr->cols * 4 ' 4 bytes per float
    
    ' Determine which pool to check
    IF size <= SMALL_MATRIX_THRESHOLD THEN
        pool_head = @g_small_pool_head
    ELSEIF size <= MEDIUM_MATRIX_THRESHOLD THEN
        pool_head = @g_medium_pool_head
    ELSEIF size <= LARGE_MATRIX_THRESHOLD THEN
        pool_head = @g_large_pool_head
    ELSE
        ' For very large matrices, just delete them
        DELETE matrix_ptr
        TrackDeallocation(size + SIZEOF(Matrix))
        g_matrix_frees = g_matrix_frees + 1
        RETURN
    END IF
    
    ' Find the entry in the pool
    entry = *pool_head
    prev = NULL
    
    WHILE entry <> NULL
        IF entry->matrix_ptr = matrix_ptr THEN
            ' Found the entry, mark as not in use
            entry->in_use = 0
            
            ' Zero the matrix data for security
            ZeroMatrix(*(entry->matrix_ptr))
            
            RETURN
        END IF
        
        prev = entry
        entry = entry->next
    WEND
    
    ' If we get here, the matrix was not found in the pool
    ' This should not happen unless the matrix was not from the pool
    PRINT "WARNING: Matrix not found in pool during return"
    DELETE matrix_ptr
    TrackDeallocation(size + SIZEOF(Matrix))
    g_matrix_frees = g_matrix_frees + 1
END SUB

' Free all entries in a matrix pool
SUB FreeMatrixPool(BYREF pool_head AS MatrixPoolEntry PTR)
    DIM entry AS MatrixPoolEntry PTR, next_entry AS MatrixPoolEntry PTR
    
    entry = pool_head
    
    WHILE entry <> NULL
        next_entry = entry->next
        
        ' Free the actual matrix
        FreeMatrix(*(entry->matrix_ptr))
        DELETE entry->matrix_ptr
        
        ' Track deallocation
        TrackDeallocation(entry->size + SIZEOF(MatrixPoolEntry) + SIZEOF(Matrix))
        g_matrix_frees = g_matrix_frees + 1
        
        ' Free the pool entry
        DELETE entry
        
        entry = next_entry
    WEND
    
    pool_head = NULL
END SUB

' Enhanced matrix allocation that uses pooling
SUB AllocateMatrix(BYREF mat AS Matrix, rows AS INTEGER, cols AS INTEGER)
    DIM matrix_ptr AS Matrix PTR
    
    ' Get a pooled matrix
    matrix_ptr = GetPooledMatrix(rows, cols)
    
    ' Copy the matrix data to the output parameter
    mat.rows = matrix_ptr->rows
    mat.cols = matrix_ptr->cols
    mat.data = matrix_ptr->data
    
    ' Note: This leaves matrix_ptr->data pointing to the same memory
    ' This is intentional - we're just copying the pointer, not the data
    ' When ReturnMatrixToPool is called, it will reference this same memory
END SUB

' Enhanced matrix deallocation that uses pooling
SUB DeallocateMatrix(BYREF mat AS Matrix)
    DIM matrix_ptr AS Matrix PTR
    
    ' Create a temporary matrix structure to hold the data
    matrix_ptr = NEW Matrix
    matrix_ptr->rows = mat.rows
    matrix_ptr->cols = mat.cols
    matrix_ptr->data = mat.data
    
    ' Return the matrix to the pool
    ReturnMatrixToPool(matrix_ptr)
    
    ' Clear the original matrix structure
    mat.rows = 0
    mat.cols = 0
    mat.data = NULL
END SUB

' *******************************************************
' * Parameter Streaming System                          *
' *******************************************************

' Set the model path for parameter streaming
SUB SetModelPath(path AS STRING)
    g_model_path = path
    PRINT "Model path set to: "; g_model_path
END SUB

' Create a layer cache entry
FUNCTION CreateLayerCache(layer_idx AS INTEGER) AS LayerCache PTR
    DIM cache AS LayerCache PTR
    cache = NEW LayerCache
    
    cache->layer_idx = layer_idx
    cache->is_loaded = 0
    cache->size = 0
    cache->last_used = TIMER
    cache->next = NULL
    
    RETURN cache
END FUNCTION

' Find a layer in the cache
FUNCTION FindLayerCache(layer_idx AS INTEGER) AS LayerCache PTR
    DIM cache AS LayerCache PTR
    
    cache = g_layer_cache_head
    
    WHILE cache <> NULL
        IF cache->layer_idx = layer_idx THEN
            ' Update last used time
            cache->last_used = TIMER
            RETURN cache
        END IF
        
        cache = cache->next
    WEND
    
    ' Not found
    RETURN NULL
END FUNCTION

' Find the least recently used layer in the cache
FUNCTION FindLRULayer() AS LayerCache PTR
    DIM cache AS LayerCache PTR, lru AS LayerCache PTR
    DIM oldest_time AS DOUBLE
    
    cache = g_layer_cache_head
    lru = cache
    
    IF cache <> NULL THEN
        oldest_time = cache->last_used
    ELSE
        RETURN NULL
    END IF
    
    WHILE cache <> NULL
        IF cache->last_used < oldest_time THEN
            oldest_time = cache->last_used
            lru = cache
        END IF
        
        cache = cache->next
    WEND
    
    RETURN lru
END FUNCTION

' Load layer weights from disk
SUB LoadLayerFromDisk(BYREF cache AS LayerCache PTR, layer_idx AS INTEGER)
    DIM filename AS STRING
    DIM file AS LONG
    DIM i AS INTEGER, num_weights AS INTEGER
    DIM rows AS INTEGER, cols AS INTEGER
    
    ' Generate the layer filename
    filename = g_model_path + "/layer_" + LTRIM(STR(layer_idx)) + ".bin"
    
    ' Open the file
    file = FREEFILE
    OPEN filename FOR BINARY AS file
    
    IF LOF(file) = 0 THEN
        PRINT "ERROR: Layer file is empty: "; filename
        CLOSE file
        RETURN
    END IF
    
    ' Read the number of weight matrices in this layer
    GET #file, , num_weights
    
    ' Allocate array for weight matrices
    REDIM cache->weights(1 TO num_weights) AS Matrix
    
    ' Track the total size of this layer
    cache->size = 0
    
    ' Read each weight matrix
    FOR i = 1 TO num_weights
        ' Read matrix dimensions
        GET #file, , rows
        GET #file, , cols
        
        ' Allocate the matrix
        AllocateMatrix(cache->weights(i), rows, cols)
        
        ' Track memory usage for this matrix
        cache->size = cache->size + (rows * cols * 4) ' 4 bytes per float
        
        ' Read matrix data directly into our allocated memory
        FOR r = 0 TO rows - 1
            FOR c = 0 TO cols - 1
                GET #file, , cache->weights(i).data(r, c)
            NEXT c
        NEXT r
    NEXT i
    
    ' Close the file
    CLOSE file
    
    ' Update cache entry
    cache->is_loaded = 1
    cache->last_used = TIMER
    
    ' Track disk read
    g_disk_reads = g_disk_reads + 1
    
    PRINT "Loaded layer "; layer_idx; " from disk. Size: "; cache->size / 1024; "KB"
END SUB

' Save layer weights to disk
SUB SaveLayerToDisk(cache AS LayerCache PTR)
    DIM filename AS STRING
    DIM file AS LONG
    DIM i AS INTEGER, num_weights AS INTEGER
    
    ' Only save if loaded and modified
    IF NOT cache->is_loaded THEN
        RETURN
    END IF
    
    ' Generate the layer filename
    filename = g_model_path + "/layer_" + LTRIM(STR(cache->layer_idx)) + ".bin"
    
    ' Open the file
    file = FREEFILE
    OPEN filename FOR BINARY AS file
    
    ' Write the number of weight matrices in this layer
    num_weights = UBOUND(cache->weights)
    PUT #file, , num_weights
    
    ' Write each weight matrix
    FOR i = 1 TO num_weights
        ' Write matrix dimensions
        PUT #file, , cache->weights(i).rows
        PUT #file, , cache->weights(i).cols
        
        ' Write matrix data
        FOR r = 0 TO cache->weights(i).rows - 1
            FOR c = 0 TO cache->weights(i).cols - 1
                PUT #file, , cache->weights(i).data(r, c)
            NEXT c
        NEXT r
    NEXT i
    
    ' Close the file
    CLOSE file
    
    ' Track disk write
    g_disk_writes = g_disk_writes + 1
    
    PRINT "Saved layer "; cache->layer_idx; " to disk."
END SUB

' Free a layer cache entry
SUB FreeLayerCache(BYREF cache AS LayerCache PTR)
    DIM i AS INTEGER
    
    ' Save to disk if needed - could add a "dirty" flag later
    SaveLayerToDisk(cache)
    
    ' Free all weight matrices
    FOR i = 1 TO UBOUND(cache->weights)
        DeallocateMatrix(cache->weights(i))
    NEXT i
    
    ' Track memory deallocation
    TrackDeallocation(cache->size)
    
    ' Free the array
    ERASE cache->weights
    
    ' Free the cache entry
    DELETE cache
    cache = NULL
    
    ' Update cache count
    g_layer_cache_count = g_layer_cache_count - 1
END SUB

' Free all layer cache entries
SUB FreeAllLayerCache()
    DIM cache AS LayerCache PTR, next_cache AS LayerCache PTR
    
    cache = g_layer_cache_head
    
    WHILE cache <> NULL
        next_cache = cache->next
        FreeLayerCache(cache)
        cache = next_cache
    WEND
    
    g_layer_cache_head = NULL
    g_layer_cache_count = 0
END SUB

' Make room in the cache for a new layer if needed
SUB EnsureCacheSpace(needed_size AS LONG)
    DIM lru AS LayerCache PTR, prev AS LayerCache PTR, current AS LayerCache PTR
    
    ' If we have room, no need to evict
    IF CanAllocate(needed_size) THEN
        RETURN
    END IF
    
    ' Keep evicting layers until we have enough space
    WHILE NOT CanAllocate(needed_size) AND g_layer_cache_count > 0
        ' Find the least recently used layer
        lru = FindLRULayer()
        
        IF lru = NULL THEN
            ' No layers to evict, this shouldn't happen
            PRINT "ERROR: No layers to evict but need more space"
            RETURN
        END IF
        
        ' Find the layer in the linked list so we can update pointers
        current = g_layer_cache_head
        prev = NULL
        
        WHILE current <> NULL AND current <> lru
            prev = current
            current = current->next
        WEND
        
        ' Update the linked list
        IF current = g_layer_cache_head THEN
            ' Removing the head
            g_layer_cache_head = current->next
        ELSE
            ' Removing from middle or end
            IF prev <> NULL THEN
                prev->next = current->next
            END IF
        END IF
        
        ' Free the layer
        PRINT "Evicting layer "; lru->layer_idx; " to make room. Size: "; lru->size / 1024; "KB"
        FreeLayerCache(lru)
    WEND
    
    ' If we still don't have enough space, that's a problem
    IF NOT CanAllocate(needed_size) THEN
        PRINT "ERROR: Cannot allocate "; needed_size / 1024; "KB even after evicting layers"
        PRINT "Try increasing MAX_MEMORY_USAGE or reducing model size"
    END IF
END SUB

' Add a layer to the cache
SUB AddLayerToCache(BYREF cache AS LayerCache PTR)
    ' Add to the head of the list
    cache->next = g_layer_cache_head
    g_layer_cache_head = cache
    
    ' Update cache count
    g_layer_cache_count = g_layer_cache_count + 1
END SUB

' Get access to a layer's weights (loading from disk if needed)
FUNCTION GetLayerWeights(layer_idx AS INTEGER, BYREF success AS INTEGER) AS LayerCache PTR
    DIM cache AS LayerCache PTR
    
    success = 0 ' Default to failure
    
    ' First check if the layer is already in the cache
    cache = FindLayerCache(layer_idx)
    
    IF cache <> NULL THEN
        ' Layer found in cache
        g_cache_hits = g_cache_hits + 1
        success = 1
        RETURN cache
    END IF
    
    ' Layer not in cache, need to load it
    g_cache_misses = g_cache_misses + 1
    
    ' Create a new cache entry
    cache = CreateLayerCache(layer_idx)
    
    ' Ensure we have enough space
    ' This will evict layers if needed
    ' Note: We don't know the exact size until we load, so estimate
    ' Could be improved by storing metadata about layer sizes
    EnsureCacheSpace(4 * 1024 * 1024) ' Assume 4MB per layer
    
    ' Now load the layer from disk
    LoadLayerFromDisk(cache, layer_idx)
    
    ' If successful, add to cache
    IF cache->is_loaded THEN
        AddLayerToCache(cache)
        success = 1
    ELSE
        ' Failed to load
        DELETE cache
        cache = NULL
    END IF
    
    RETURN cache
END FUNCTION

' *******************************************************
' * Memory-Aware Matrix Operations                      *
' *******************************************************

' Perform a matrix multiplication while considering memory constraints
SUB MemoryAwareMatrixMultiply(A AS Matrix, B AS Matrix, BYREF C AS Matrix)
    DIM rows AS INTEGER, cols AS INTEGER, inner AS INTEGER
    DIM needed_memory AS LONG
    
    ' Calculate dimensions
    rows = A.rows
    inner = A.cols
    cols = B.cols
    
    ' Ensure B matrix is compatible
    IF inner <> B.rows THEN
        PRINT "ERROR: Matrix dimensions incompatible for multiplication"
        RETURN
    END IF
    
    ' Calculate memory needed for result matrix
    needed_memory = rows * cols * 4 ' 4 bytes per float
    
    ' Check if we have enough memory
    IF NOT CanAllocate(needed_memory) THEN
        PRINT "WARNING: Not enough memory for result matrix. Attempting to free space..."
        
        ' Try to free some space
        EnsureCacheSpace(needed_memory)
        
        ' Check again
        IF NOT CanAllocate(needed_memory) THEN
            PRINT "ERROR: Cannot allocate memory for matrix multiplication result"
            RETURN
        END IF
    END IF
    
    ' Allocate result matrix
    AllocateMatrix(C, rows, cols)
    
    ' Use existing matrix multiply function
    MatrixMultiply(A, B, C)
END SUB

' Perform block-sparse attention with memory constraints
SUB MemoryAwareBlockSparseAttention(query AS Matrix, key AS Matrix, value AS Matrix, BYREF output AS Matrix, use_causal_mask AS INTEGER)
    ' First check if we have enough memory for the QK^T matrix
    ' This is usually the largest temporary allocation
    DIM qkt_memory AS LONG
    qkt_memory = query.rows * key.rows * 4 ' 4 bytes per float for QK^T
    
    ' Check if we can do it directly
    IF CanAllocate(qkt_memory) THEN
        ' We have enough memory, use standard implementation
        BlockSparseAttention(query, key, value, output, use_causal_mask)
        RETURN
    END IF
    
    ' Not enough memory, need to use a streaming approach with smaller blocks
    PRINT "Using block streaming for attention (memory constrained)..."
    
    ' Determine block size based on available memory
    ' Leave 25% of available memory for other operations
    DIM available AS LONG = GetAvailableMemory() * 0.75
    DIM block_rows AS INTEGER
    
    ' Calculate max rows we can process at once
    block_rows = INT(available / (key.rows * 4))
    
    ' Ensure at least 1 row
    IF block_rows < 1 THEN block_rows = 1
    
    ' Initialize output matrix
    InitMatrix(output, query.rows, value.cols)
    ZeroMatrix(output)
    
    ' Process attention in blocks
    DIM start_row AS INTEGER, end_row AS INTEGER
    DIM query_block AS Matrix, output_block AS Matrix
    
    FOR start_row = 0 TO query.rows - 1 STEP block_rows
        ' Calculate end row for this block
        end_row = start_row + block_rows - 1
        IF end_row >= query.rows THEN end_row = query.rows - 1
        
        ' Extract query block
        InitMatrix(query_block, end_row - start_row + 1, query.cols)
        FOR i = 0 TO query_block.rows - 1
            FOR j = 0 TO query_block.cols - 1
                query_block.data(i, j) = query.data(start_row + i, j)
            NEXT j
        NEXT i
        
        ' Process this block
        BlockSparseAttention(query_block, key, value, output_block, use_causal_mask)
        
        ' Copy results to output matrix
        FOR i = 0 TO output_block.rows - 1
            FOR j = 0 TO output_block.cols - 1
                output.data(start_row + i, j) = output_block.data(i, j)
            NEXT j
        NEXT i
        
        ' Free temporary matrices
        FreeMatrix(query_block)
        FreeMatrix(output_block)
    NEXT start_row
END SUB

' *******************************************************
' * Testing Functions                                   *
' *******************************************************

' Test the memory manager with a simulated model
SUB TestMemoryManager()
    DIM i AS INTEGER, j AS INTEGER, success AS INTEGER
    DIM layer_cache AS LayerCache PTR
    DIM start_time AS DOUBLE, end_time AS DOUBLE
    
PRINT "Testing Memory Manager with simulated transformer model..."
    PRINT "======================================================="
    
    ' Initialize memory manager
    InitMemoryManager()
    
    ' Create temporary directory for model files
    MKDIR "temp_model"
    SetModelPath("temp_model")
    
    ' Generate some simulated layer files
    PRINT "Generating test layer files..."
    
    ' Create 10 layers with different sizes
    DIM layer_size(1 TO 10) AS INTEGER
    DIM matrix_count(1 TO 10) AS INTEGER
    
    ' Varying layer sizes to test caching/eviction
    FOR i = 1 TO 10
        ' Each layer will have between 2 and 5 matrices
        matrix_count(i) = 2 + (i MOD 4)
        
        ' Simulate layer file creation
        DIM file AS LONG
        DIM filename AS STRING
        filename = g_model_path + "/layer_" + LTRIM(STR(i)) + ".bin"
        
        file = FREEFILE
        OPEN filename FOR BINARY AS file
        
        ' Write number of matrices
        PUT #file, , matrix_count(i)
        
        ' Total size for this layer
        layer_size(i) = 0
        
        ' Create random matrices of different sizes
        FOR j = 1 TO matrix_count(i)
            ' Vary sizes based on layer and matrix index
            DIM rows AS INTEGER, cols AS INTEGER
            rows = 32 + (i * 16) + (j * 8)
            cols = 32 + (i * 8) + (j * 4)
            
            ' Write dimensions
            PUT #file, , rows
            PUT #file, , cols
            
            ' Write data
            DIM val AS SINGLE
            FOR r = 0 TO rows - 1
                FOR c = 0 TO cols - 1
                    val = (r * c * i * j) / 1000.0 ' Deterministic but varies
                    PUT #file, , val
                NEXT c
            NEXT r
            
            ' Add to layer size
            layer_size(i) = layer_size(i) + (rows * cols * 4) ' 4 bytes per float
        NEXT j
        
        CLOSE file
        
        PRINT "Created layer "; i; " with "; matrix_count(i); " matrices, size: "; _
              layer_size(i) / 1024; "KB"
    NEXT i
    
    ' Test 1: Matrix Pooling
    PRINT
    PRINT "Test 1: Matrix Pooling"
    PRINT "---------------------"
    
    ' Fill pools with various sized matrices
    DIM mat1 AS Matrix, mat2 AS Matrix, mat3 AS Matrix
    DIM matrices(1 TO 100) AS Matrix
    
    ' Allocate some matrices
    start_time = TIMER
    FOR i = 1 TO 100
        ' Vary sizes
        AllocateMatrix(matrices(i), 10 + i, 10 + i)
        
        ' Fill with some values
        FOR r = 0 TO matrices(i).rows - 1
            FOR c = 0 TO matrices(i).cols - 1
                matrices(i).data(r, c) = r * c * i
            NEXT c
        NEXT r
    NEXT i
    
    ' Print stats
    PRINT "Allocated 100 matrices of various sizes"
    PRINT "Matrix allocations: "; g_matrix_allocs
    PRINT "Memory usage: "; g_current_memory_usage / 1024; "KB"
    
    ' Free half the matrices
    FOR i = 1 TO 50
        DeallocateMatrix(matrices(i))
    NEXT i
    
    PRINT "Freed 50 matrices"
    PRINT "Memory usage: "; g_current_memory_usage / 1024; "KB"
    
    ' Allocate 50 more matrices (should reuse some from pool)
    FOR i = 1 TO 50
        AllocateMatrix(matrices(i), 10 + i, 10 + i)
    NEXT i
    
    PRINT "Reallocated 50 matrices"
    PRINT "Matrix allocations: "; g_matrix_allocs
    PRINT "Cache hits: "; g_cache_hits
    
    ' Free all
    FOR i = 1 TO 100
        DeallocateMatrix(matrices(i))
    NEXT i
    
    end_time = TIMER
    PRINT "Matrix pooling test completed in "; end_time - start_time; " seconds"
    
    ' Test 2: Layer Cache with LRU Replacement
    PRINT
    PRINT "Test 2: Layer Cache and Streaming"
    PRINT "--------------------------------"
    
    ' Set low cache size to force evictions
    g_max_cached_layers = 3
    
    start_time = TIMER
    
    ' Access layers in a pattern that will cause cache evictions
    PRINT "Accessing layers in sequence to test LRU cache:"
    
    FOR test = 1 TO 2
        ' First pass should be all misses, second should have some hits
        PRINT "Pass "; test; ":"
        
        ' Access layers in various orders to test LRU
        DIM access_sequence(1 TO 15) AS INTEGER
        access_sequence(1) = 1
        access_sequence(2) = 2
        access_sequence(3) = 3
        access_sequence(4) = 4 ' Should evict layer 1
        access_sequence(5) = 1 ' Should evict layer 2
        access_sequence(6) = 5 ' Should evict layer 3
        access_sequence(7) = 3 ' Hit
        access_sequence(8) = 6 ' Should evict layer 4
        access_sequence(9) = 7 ' Should evict layer 1
        access_sequence(10) = 3 ' Hit
        access_sequence(11) = 5 ' Hit
        access_sequence(12) = 8 ' Should evict layer 6
        access_sequence(13) = 9 ' Should evict layer 7
        access_sequence(14) = 10 ' Should evict layer 3
        access_sequence(15) = 5 ' Hit
        
        FOR i = 1 TO 15
            ' Access a layer
            layer_cache = GetLayerWeights(access_sequence(i), success)
            
            ' Check if access was successful
            IF success THEN
                PRINT "  Accessed layer "; access_sequence(i); ": ";
                IF layer_cache->weights(1).rows > 0 THEN
                    PRINT "OK ("; layer_cache->size / 1024; "KB)"
                ELSE
                    PRINT "ERROR: Layer data not loaded"
                END IF
            ELSE
                PRINT "  FAILED to access layer "; access_sequence(i)
            END IF
            
            ' Print memory stats periodically
            IF i MOD 5 = 0 THEN
                PRINT "  Memory usage: "; g_current_memory_usage / 1024; "KB, Cache hits: "; g_cache_hits
            END IF
        NEXT i
    NEXT test
    
    end_time = TIMER
    PRINT "Layer cache test completed in "; end_time - start_time; " seconds"
    PRINT "Cache hits: "; g_cache_hits; ", Misses: "; g_cache_misses
    PRINT "Disk reads: "; g_disk_reads; ", Disk writes: "; g_disk_writes
    
    ' Test 3: Memory-aware matrix operations
    PRINT
    PRINT "Test 3: Memory-aware Operations"
    PRINT "------------------------------"
    
    ' Create test matrices
    DIM big_mat1 AS Matrix, big_mat2 AS Matrix, result_mat AS Matrix
    DIM size AS INTEGER
    
    start_time = TIMER
    
    ' Try different sizes to test memory adaptation
    DIM sizes(1 TO 3) AS INTEGER
    sizes(1) = 128  ' Small
    sizes(2) = 512  ' Medium
    sizes(3) = 1024 ' Large - should require memory adaptation
    
    FOR t = 1 TO 3
        size = sizes(t)
        
        PRINT "Testing with matrices of size "; size; "x"; size
        
        ' Create test matrices
        InitMatrix(big_mat1, size, size)
        InitMatrix(big_mat2, size, size)
        
        ' Fill with values
        FOR i = 0 TO size - 1
            FOR j = 0 TO size - 1
                big_mat1.data(i, j) = (i * j) / 1000.0
                big_mat2.data(i, j) = (i + j) / 1000.0
            NEXT j
        NEXT i
        
        ' Memory stats before
        PRINT "  Before operation: "; g_current_memory_usage / 1024; "KB used, "; _
              GetAvailableMemory() / 1024; "KB available"
        
        ' Perform memory-aware multiplication
        PRINT "  Performing matrix multiplication..."
        
        MemoryAwareMatrixMultiply(big_mat1, big_mat2, result_mat)
        
        ' Memory stats after
        PRINT "  After operation: "; g_current_memory_usage / 1024; "KB used, "; _
              GetAvailableMemory() / 1024; "KB available"
        PRINT "  Result matrix: "; result_mat.rows; "x"; result_mat.cols
        
        ' Verify a few values
        PRINT "  Verification: result[0,0]="; result_mat.data(0, 0); _
              ", result[mid,mid]="; result_mat.data(size\2, size\2)
        
        ' Clean up
        FreeMatrix(big_mat1)
        FreeMatrix(big_mat2)
        FreeMatrix(result_mat)
    NEXT t
    
    end_time = TIMER
    PRINT "Memory-aware operations test completed in "; end_time - start_time; " seconds"
    
    ' Clean up
    PRINT
    PRINT "Cleaning up..."
    
    ' Shutdown memory manager
    ShutdownMemoryManager()
    
    ' Print final memory stats
    PrintMemoryStats()
    
    ' Cleanup test files
    PRINT "Removing test files..."
    SYSTEM("DEL /Q temp_model\\*.*")
    SYSTEM("RMDIR temp_model")
    
    PRINT "Memory manager test complete!"
END SUB

' Main entry point for memory manager testing
SUB TestMemoryManager_Main()
    PRINT "GPT-2 BASIC Memory Manager Test"
    PRINT "================================"
    PRINT
    
    ' Run the test
    TestMemoryManager()
END SUB
