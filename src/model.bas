' BASIC implementation of the full GPT-2-like model.
' This file assembles the transformer components, handles tokenization,
' embedding, and the text generation loop.

' Include necessary files
#INCLUDE "data_structures.bas"
#INCLUDE "quantization.bas"
#INCLUDE "matrix_ops.bas"
#INCLUDE "transformer_components.bas"
#INCLUDE "softmax_fixed.bas" ' For fixed-point softmax implementation
#INCLUDE "block_sparse.bas" ' For block-sparse attention optimization
#INCLUDE "file_io.bas" ' For streaming model parameters from disk
#INCLUDE "tokenizer.bas" ' For tokenization and vocabulary management

' Default model architecture parameters
CONST DEFAULT_EMBEDDING_DIM AS INTEGER = 128
CONST DEFAULT_NUM_HEADS AS INTEGER = 4
CONST DEFAULT_CONTEXT_LENGTH AS INTEGER = 128
CONST DEFAULT_NUM_LAYERS AS INTEGER = 2
CONST DEFAULT_VOCAB_SIZE AS INTEGER = 1000
CONST DEFAULT_INTERMEDIATE_DIM AS INTEGER = DEFAULT_EMBEDDING_DIM * 4 ' Typically 4x embedding dimension

' GPT-2 Model structure definition
TYPE GPT2Model
    ' Embedding layer weights
    token_embed AS Matrix ' Token embeddings (vocab_size, embedding_dim)
    pos_embed AS Matrix   ' Positional embeddings (context_length, embedding_dim)

    ' Layer weights - dynamically allocated based on layer count
    attention_qkv(ANY) AS Matrix ' Combined query-key-value projections for efficiency
    attention_output(ANY) AS Matrix ' Output projections
    ffn_up(ANY) AS Matrix   ' FFN up projections (W1)
    ffn_down(ANY) AS Matrix ' FFN down projections (W2)
    ffn_gate(ANY) AS Matrix ' FFN gate projections (W3)

    ' Layer normalization parameters
    ln1_gamma(ANY) AS Matrix ' First LN gamma parameters
    ln1_beta(ANY) AS Matrix  ' First LN beta parameters
    ln2_gamma(ANY) AS Matrix ' Second LN gamma parameters
    ln2_beta(ANY) AS Matrix  ' Second LN beta parameters

    ' Final layer norm weights
    final_ln_gamma AS Matrix ' (embedding_dim, 1)
    final_ln_beta AS Matrix  ' (embedding_dim, 1)

    ' Output layer weights - may use token embedding tied weights
    output_proj AS Matrix ' (embedding_dim, vocab_size) or NULL if tied weights

    ' Memory management
    streaming_mode AS INTEGER ' 0: in-memory, 1: streaming from disk
    file_info AS ModelFileInfo ' File handles for streaming mode

    ' Currently active layer for memory management
    active_layer AS INTEGER
END TYPE

' Global vocabulary mapping (token ID to string)
DIM SHARED g_vocabulary() AS STRING

' Current active layer for memory management
DIM SHARED g_active_layer AS INTEGER = -1

' Global model instance
DIM SHARED g_model AS GPT2Model

' Load model configuration from file
FUNCTION LoadTextModelConfig(BYREF config AS ModelConfig, config_path AS STRING) AS INTEGER
    DIM file AS LONG
    DIM success AS INTEGER

    ' Try to open the config file
    success = 0
    file = FREEFILE
    ON ERROR GOTO config_error
    OPEN config_path FOR INPUT AS #file
    success = 1

    ' Parse the config file (simple key=value format)
    DIM config_line AS STRING
    DIM key AS STRING, value AS STRING
    DIM eq_pos AS INTEGER

    ' Set defaults first
    InitModelConfig(config)

    WHILE EOF(file) = 0
        LINE INPUT #file, config_line
        config_line = TRIM$(config_line)

        ' Skip comments and empty lines
        IF LEFT$(config_line, 1) = "#" OR config_line = "" THEN CONTINUE WHILE

        ' Parse key=value
        eq_pos = INSTR(config_line, "=")
        IF eq_pos > 0 THEN
            key = TRIM$(LEFT$(config_line, eq_pos - 1))
            value = TRIM$(MID$(config_line, eq_pos + 1))

            ' Set config values based on key
            SELECT CASE LCASE$(key)
                CASE "n_embd", "embedding_dim"
                    config.n_embd = VAL(value)
                CASE "n_head", "num_heads"
                    config.n_head = VAL(value)
                CASE "n_layer", "num_layers"
                    config.n_layer = VAL(value)
                CASE "n_positions", "context_length"
                    config.n_positions = VAL(value)
                CASE "vocab_size"
                    config.vocab_size = VAL(value)
            END SELECT
        END IF
    WEND

    CLOSE #file
    ON ERROR GOTO 0
    LoadTextModelConfig = 1
    EXIT FUNCTION

config_error:
    IF success <> 0 THEN CLOSE #file
    ON ERROR GOTO 0
    LoadTextModelConfig = 0
    EXIT FUNCTION
END FUNCTION

' Initialize the GPT-2 model
FUNCTION InitGPT2Model(BYREF model AS GPT2Model, config AS ModelConfig) AS INTEGER
    DIM i AS INTEGER, j AS INTEGER
    DIM gamma_value AS INTEGER

    ' Store global config
    g_config = config

    ' Initialize model components
    ' Token embeddings
    InitMatrix model.token_embed, config.vocab_size, config.n_embd
    ' Positional embeddings
    InitMatrix model.pos_embed, config.n_positions, config.n_embd

    ' Resize arrays for layer parameters
    REDIM model.attention_qkv(config.n_layer - 1)
    REDIM model.attention_output(config.n_layer - 1)
    REDIM model.ffn_up(config.n_layer - 1)
    REDIM model.ffn_down(config.n_layer - 1)
    REDIM model.ffn_gate(config.n_layer - 1)
    REDIM model.ln1_gamma(config.n_layer - 1)
    REDIM model.ln1_beta(config.n_layer - 1)
    REDIM model.ln2_gamma(config.n_layer - 1)
    REDIM model.ln2_beta(config.n_layer - 1)
    gamma_value = FixedToLogQuantized(FloatToFixed(1.0)).packed_value

    FOR i = 0 TO config.n_layer - 1
        InitMatrix model.attention_qkv(i), config.n_embd, config.n_embd
        InitMatrix model.attention_output(i), config.n_embd, config.n_embd
        InitMatrix model.ffn_up(i), config.n_embd, config.n_embd * 4
        InitMatrix model.ffn_down(i), config.n_embd * 4, config.n_embd
        InitMatrix model.ffn_gate(i), config.n_embd, config.n_embd * 4
        InitMatrix model.ln1_gamma(i), config.n_embd, 1
        InitMatrix model.ln1_beta(i), config.n_embd, 1
        InitMatrix model.ln2_gamma(i), config.n_embd, 1
        InitMatrix model.ln2_beta(i), config.n_embd, 1

        FOR j = 0 TO config.n_embd - 1
            model.ln1_gamma(i).data(j, 0) = gamma_value
            model.ln2_gamma(i).data(j, 0) = gamma_value
        NEXT j
    NEXT i

    ' Initialize final layer norm parameters
    InitMatrix model.final_ln_gamma, config.n_embd, 1
    InitMatrix model.final_ln_beta, config.n_embd, 1
    FOR j = 0 TO config.n_embd - 1
        model.final_ln_gamma.data(j, 0) = gamma_value
    NEXT j

    ' Initialize output projection
    InitMatrix model.output_proj, config.n_embd, config.vocab_size

    ' Set model to initially use in-memory mode
    model.streaming_mode = 0
    model.active_layer = -1

    RETURN 1
END FUNCTION

' Load model parameters from a directory
FUNCTION LoadModelFromPath(BYREF model AS GPT2Model, model_dir AS STRING) AS INTEGER
    DIM success AS INTEGER

    PRINT "Loading model from: "; model_dir

    IF ModelDirectoryHasGPT2BasicCheckpoint(model_dir) <> 0 THEN
        IF GPT2BasicLoadModel(model_dir) <> 0 THEN
            g_config.n_embd = GPT2BasicEmbeddingDim()
            g_config.n_head = GPT2BasicHeadCount()
            g_config.n_layer = GPT2BasicLayerCount()
            g_config.n_positions = GPT2BasicContextLength()
            g_config.vocab_size = GPT2BasicVocabSize()

            InitModelFiles model.file_info, model_dir
            model.streaming_mode = 1
            model.active_layer = -1
            g_active_layer = -1

            PRINT "Loaded GPT2-BASIC checkpoint:"
            PRINT "  Profile:    "; GPT2BasicProfileName()
            PRINT "  Layers:     "; g_config.n_layer
            PRINT "  Embed dim:  "; g_config.n_embd
            PRINT "  Heads:      "; g_config.n_head
            PRINT "  Context:    "; g_config.n_positions
            PRINT "  Vocab size: "; g_config.vocab_size
            RETURN 1
        END IF
    END IF

    ' Check if directory exists
    DIM dir_check AS INTEGER
    dir_check = FREEFILE

    ON ERROR GOTO dir_error
    OPEN model_dir + "\CONFIG.BIN" FOR INPUT AS #dir_check
    CLOSE #dir_check
    success = 1

    ' Initialize model file info with the directory path
    InitModelFiles model.file_info, model_dir

    ' Load model configuration
    DIM cfg_embedding_dim AS INTEGER, cfg_num_heads AS INTEGER
    DIM cfg_num_layers AS INTEGER, cfg_context_length AS INTEGER, cfg_vocab_size AS INTEGER

    LoadModelConfig model.file_info, cfg_embedding_dim, cfg_num_heads, cfg_num_layers, cfg_context_length, cfg_vocab_size

    ' Use loaded configuration to initialize the model structure
    g_config.n_embd = cfg_embedding_dim
    g_config.n_head = cfg_num_heads
    g_config.n_layer = cfg_num_layers
    g_config.n_positions = cfg_context_length
    g_config.vocab_size = cfg_vocab_size

    ' Initialize model with this configuration
    success = InitGPT2Model(model, g_config)
    IF success = 0 THEN GOTO load_error

    ' Set streaming mode
    model.streaming_mode = 1

    ' Load token embeddings for immediate use (always keep in memory)
    LoadMatrix model.file_info.token_embed_file, model.token_embed
    LoadMatrix model.file_info.pos_embed_file, model.pos_embed

    ' Load vocabulary
    REDIM g_vocabulary(g_config.vocab_size - 1)
    LoadModelVocabulary model.file_info, g_vocabulary()

    PRINT "Model configuration loaded:"
    PRINT "  Embedding dimension: "; g_config.n_embd
    PRINT "  Number of heads:     "; g_config.n_head
    PRINT "  Number of layers:    "; g_config.n_layer
    PRINT "  Context length:      "; g_config.n_positions
    PRINT "  Vocabulary size:     "; g_config.vocab_size
    PRINT "  Using streaming mode: YES"

    GOTO load_done

dir_error:
    PRINT "Error: Model directory not found or inaccessible: "; model_dir
    success = 0
    IF dir_check <> 0 THEN CLOSE #dir_check
    GOTO load_done

load_error:
    PRINT "Error: Failed to initialize model"
    success = 0

load_done:
    ON ERROR GOTO 0
    RETURN success
END FUNCTION

' Main model loading function
FUNCTION LoadModel(filepath AS STRING) AS INTEGER
    ' Set up the model path based on whether it's a directory or file
    DIM last_char AS STRING * 1
    DIM success AS INTEGER
    DIM model_path AS STRING
    DIM had_trailing_separator AS INTEGER

    model_path = filepath
    last_char = RIGHT$(model_path, 1)

    had_trailing_separator = 0
    IF last_char = "/" OR last_char = "\" THEN
        had_trailing_separator = 1
        model_path = LEFT$(model_path, LEN(model_path) - 1)
        last_char = RIGHT$(model_path, 1)
    END IF

    IF ModelDirectoryHasGPT2BasicCheckpoint(model_path) <> 0 THEN
        RETURN LoadModelFromPath(g_model, model_path)
    END IF

    IF had_trailing_separator <> 0 THEN
        ' It's a directory path - use streaming mode
        success = LoadModelFromPath(g_model, filepath)
    ELSE
        ' For single file models (not implemented yet)
        PRINT "Error: Single file models not supported. Please provide a directory path."
        success = 0
    END IF

    RETURN success
END FUNCTION

' Clean up model resources and free memory
SUB FreeGPT2Model(BYREF model AS GPT2Model)
    DIM i AS INTEGER

    IF GPT2BasicIsLoaded() <> 0 THEN
        GPT2BasicFreeModel()
        model.streaming_mode = 0
        model.active_layer = -1
        g_active_layer = -1
        RETURN
    END IF

    ' Free embedding matrices
    FreeMatrix model.token_embed
    FreeMatrix model.pos_embed

    ' Free layer parameters that are in memory
    FOR i = 0 TO g_config.n_layer - 1
        ' Only free if this layer was loaded or if in non-streaming mode
        IF i = model.active_layer OR model.streaming_mode = 0 THEN
            IF model.attention_qkv(i).rows > 0 THEN FreeMatrix model.attention_qkv(i)
            IF model.attention_output(i).rows > 0 THEN FreeMatrix model.attention_output(i)
            IF model.ffn_up(i).rows > 0 THEN FreeMatrix model.ffn_up(i)
            IF model.ffn_down(i).rows > 0 THEN FreeMatrix model.ffn_down(i)
            IF model.ffn_gate(i).rows > 0 THEN FreeMatrix model.ffn_gate(i)
            IF model.ln1_gamma(i).rows > 0 THEN FreeMatrix model.ln1_gamma(i)
            IF model.ln1_beta(i).rows > 0 THEN FreeMatrix model.ln1_beta(i)
            IF model.ln2_gamma(i).rows > 0 THEN FreeMatrix model.ln2_gamma(i)
            IF model.ln2_beta(i).rows > 0 THEN FreeMatrix model.ln2_beta(i)
        END IF
    NEXT i

    ' Free final layer norm and output projection
    FreeMatrix model.final_ln_gamma
    FreeMatrix model.final_ln_beta
    FreeMatrix model.output_proj

    ' Close all file handles if in streaming mode
    IF model.streaming_mode = 1 THEN
        CloseAllFiles model.file_info
    END IF

    ' Reset active layer tracker
    model.active_layer = -1
    g_active_layer = -1
END SUB

' For backward compatibility - calls the new function with global model
SUB FreeModelResources()
    ' Free the global model
    FreeGPT2Model(g_model)
END SUB

' Integration with the tokenizer module
' This simply wraps the tokenizer module's Encode routine.
SUB TokenizeText(input_text AS STRING, tokens() AS INTEGER, BYREF token_count AS INTEGER)
    Encode input_text, tokens(), token_count
END SUB

' Function to perform embedding lookup and add positional encoding.
' Input: Sequence of token IDs
' Output: Embedded and positionally encoded matrix (context_length, embedding_dim)
SUB EmbedTokens (model AS GPT2Model, tokens() AS INTEGER, BYREF output_mat AS Matrix)
    DIM i AS INTEGER ' Token index (position in sequence)
    DIM j AS INTEGER ' Embedding dimension index
    DIM token_id AS INTEGER
    DIM context_len AS INTEGER

    ' Get actual context length from the number of tokens
    context_len = UBOUND(tokens) + 1
    IF context_len > g_config.n_positions THEN
        context_len = g_config.n_positions
    END IF

    ' Initialize output matrix with the right dimensions
    IF output_mat.rows <> context_len OR output_mat.cols <> g_config.n_embd THEN
        FreeMatrix output_mat
        InitMatrix output_mat, context_len, g_config.n_embd
    END IF

    ' Process each token position
    FOR i = 0 TO context_len - 1
        token_id = tokens(i)

        ' Ensure token_id is valid
        IF token_id < 0 OR token_id >= g_config.vocab_size THEN
            token_id = 0 ' Use the first token as UNK
        END IF

        ' Copy token embedding vector
        FOR j = 0 TO g_config.n_embd - 1
            output_mat.data(i, j) = model.token_embed.data(token_id, j)
        NEXT j

        ' Add positional embedding vector
        FOR j = 0 TO g_config.n_embd - 1
            ' Convert to fixed-point, add, then convert back
            DIM fp_embed AS INTEGER = DequantizeLogToFixed(output_mat.data(i, j))
            DIM fp_pos_embed AS INTEGER = DequantizeLogToFixed(model.pos_embed.data(i, j))
            DIM fp_sum AS INTEGER = FixedAdd(fp_embed, fp_pos_embed)
            output_mat.data(i, j) = FixedToLogQuantized(fp_sum).packed_value
        NEXT j
    NEXT i
END SUB

' Function to perform a forward pass through a single transformer layer.
' Input: Input matrix (context_length, embedding_dim)
' Output: Output matrix (context_length, embedding_dim)
SUB TransformerLayer(model AS GPT2Model, input_mat AS Matrix, layer_idx AS INTEGER, BYREF output_mat AS Matrix)
    ' Assumes Input is pre-initialized. Output will be initialized if needed.

    ' Make sure Output has correct dimensions
    IF output_mat.rows <> input_mat.rows OR output_mat.cols <> input_mat.cols THEN
        FreeMatrix output_mat
        InitMatrix output_mat, input_mat.rows, input_mat.cols
    END IF

    ' In streaming mode, load layer weights from disk when needed
    IF model.streaming_mode = 1 THEN
        ' Check if we need to load a different layer than the currently active one
        IF model.active_layer <> layer_idx THEN
            ' Free previous active layer if one was loaded
            IF model.active_layer >= 0 THEN
                ' Free the previously active layer's matrices
                IF model.attention_qkv(model.active_layer).rows > 0 THEN
                    FreeMatrix model.attention_qkv(model.active_layer)
                END IF
                IF model.attention_output(model.active_layer).rows > 0 THEN
                    FreeMatrix model.attention_output(model.active_layer)
                END IF
                IF model.ffn_up(model.active_layer).rows > 0 THEN
                    FreeMatrix model.ffn_up(model.active_layer)
                END IF
                IF model.ffn_down(model.active_layer).rows > 0 THEN
                    FreeMatrix model.ffn_down(model.active_layer)
                END IF
                IF model.ffn_gate(model.active_layer).rows > 0 THEN
                    FreeMatrix model.ffn_gate(model.active_layer)
                END IF
                IF model.ln1_gamma(model.active_layer).rows > 0 THEN
                    FreeMatrix model.ln1_gamma(model.active_layer)
                END IF
                IF model.ln1_beta(model.active_layer).rows > 0 THEN
                    FreeMatrix model.ln1_beta(model.active_layer)
                END IF
                IF model.ln2_gamma(model.active_layer).rows > 0 THEN
                    FreeMatrix model.ln2_gamma(model.active_layer)
                END IF
                IF model.ln2_beta(model.active_layer).rows > 0 THEN
                    FreeMatrix model.ln2_beta(model.active_layer)
                END IF
            END IF

            ' Initialize matrices for the new layer
            InitMatrix model.attention_qkv(layer_idx), g_config.n_embd, g_config.n_embd * 3
            InitMatrix model.attention_output(layer_idx), g_config.n_embd, g_config.n_embd
            InitMatrix model.ffn_up(layer_idx), g_config.n_embd, g_config.n_embd * 4
            InitMatrix model.ffn_down(layer_idx), g_config.n_embd * 4, g_config.n_embd
            InitMatrix model.ffn_gate(layer_idx), g_config.n_embd, g_config.n_embd * 4
            InitMatrix model.ln1_gamma(layer_idx), g_config.n_embd, 1
            InitMatrix model.ln1_beta(layer_idx), g_config.n_embd, 1
            InitMatrix model.ln2_gamma(layer_idx), g_config.n_embd, 1
            InitMatrix model.ln2_beta(layer_idx), g_config.n_embd, 1

            ' Stream layer weights for this layer
            ' For backward compatibility, we'll structure this to work with the old system
            ' when streaming from disk
            StreamLayerWeights model.file_info, layer_idx, _
                               model.attention_qkv(layer_idx), model.attention_qkv(layer_idx), model.attention_qkv(layer_idx), model.attention_output(layer_idx), _
                               model.ffn_up(layer_idx), model.ffn_down(layer_idx), model.ffn_gate(layer_idx), _
                               model.ln1_gamma(layer_idx), model.ln1_beta(layer_idx), _
                               model.ln2_gamma(layer_idx), model.ln2_beta(layer_idx)

            ' Update active layer tracker
            model.active_layer = layer_idx
            g_active_layer = layer_idx
        END IF
    END IF

    ' Need temporary matrices for residual connections and intermediate results
    DIM Residual AS Matrix: InitMatrix Residual, input_mat.rows, input_mat.cols
    DIM Norm1Output AS Matrix: InitMatrix Norm1Output, input_mat.rows, input_mat.cols
    DIM AttentionOutput AS Matrix: InitMatrix AttentionOutput, input_mat.rows, input_mat.cols
    DIM Norm2Output AS Matrix: InitMatrix Norm2Output, input_mat.rows, input_mat.cols
    DIM FFNOutput AS Matrix: InitMatrix FFNOutput, input_mat.rows, input_mat.cols

    ' Step 1: Layer Normalization (before attention)
    LayerNorm input_mat, model.ln1_gamma(layer_idx), model.ln1_beta(layer_idx), Norm1Output

    ' Step 2: Multi-Head Attention
    ' Need to pass the correct weight matrices for this layer
    MultiHeadAttention Norm1Output, model.attention_qkv(layer_idx), model.attention_qkv(layer_idx), model.attention_qkv(layer_idx), model.attention_output(layer_idx), AttentionOutput

    ' Step 3: Add Residual Connection and Apply Layer Normalization (after attention)
    MatrixAdd input_mat, AttentionOutput, Residual

    ' Layer Normalization (after residual)
    LayerNorm Residual, model.ln2_gamma(layer_idx), model.ln2_beta(layer_idx), Norm2Output

    ' Step 4: Feed-Forward Network
    FeedForward Norm2Output, model.ffn_up(layer_idx), model.ffn_down(layer_idx), model.ffn_gate(layer_idx), FFNOutput

    ' Step 5: Add Residual Connection (after FFN)
    MatrixAdd Residual, FFNOutput, output_mat

    ' Clean up temporary matrices
    FreeMatrix Residual
    FreeMatrix Norm1Output
    FreeMatrix AttentionOutput
    FreeMatrix Norm2Output
    FreeMatrix FFNOutput
END SUB

' Function to perform the full forward pass through the model.
' Input: Model, sequence of token IDs
' Output: Logits matrix (context_length, vocab_size)
SUB ForwardPass(model AS GPT2Model, tokens() AS INTEGER, BYREF Logits AS Matrix)
    DIM context_len AS INTEGER
    DIM raw_context_len AS INTEGER
    DIM vocab_idx AS INTEGER

    ' Determine actual context length
    raw_context_len = UBOUND(tokens) + 1
    context_len = raw_context_len
    IF context_len > g_config.n_positions THEN
        context_len = g_config.n_positions
    END IF

    IF GPT2BasicIsLoaded() <> 0 THEN
        IF context_len < 1 THEN context_len = 1
        IF context_len > GPT2BasicContextLength() THEN context_len = GPT2BasicContextLength()

        IF Logits.rows <> context_len OR Logits.cols <> GPT2BasicVocabSize() THEN
            FreeMatrix Logits
            InitMatrix Logits, context_len, GPT2BasicVocabSize()
        END IF
        ZeroMatrix Logits

        IF GPT2BasicIsFixedPointLoaded() <> 0 THEN
            DIM fx_logits() AS LONG
            IF GPT2BasicForwardFixedLogits(tokens(), raw_context_len, fx_logits()) <> 0 THEN
                FOR vocab_idx = 0 TO GPT2BasicVocabSize() - 1
                    Logits.data(context_len - 1, vocab_idx) = CSNG(fx_logits(vocab_idx)) / 4096.0
                NEXT vocab_idx
            END IF
        ELSE
            DIM float_logits() AS SINGLE
            IF GPT2BasicForwardFloatLogits(tokens(), raw_context_len, float_logits()) <> 0 THEN
                FOR vocab_idx = 0 TO GPT2BasicVocabSize() - 1
                    Logits.data(context_len - 1, vocab_idx) = float_logits(vocab_idx)
                NEXT vocab_idx
            END IF
        END IF

        RETURN
    END IF

    ' Initialize Logits matrix if needed
    IF Logits.rows <> context_len OR Logits.cols <> g_config.vocab_size THEN
        FreeMatrix Logits
        InitMatrix Logits, context_len, g_config.vocab_size
    END IF

    ' Need temporary matrices for layer inputs and outputs
    DIM LayerInput AS Matrix
    DIM LayerOutput AS Matrix

    ' Step 1: Embed tokens and add positional encoding
    EmbedTokens model, tokens(), LayerInput

    ' Initialize output matrix for first layer
    InitMatrix LayerOutput, LayerInput.rows, LayerInput.cols

    ' Step 2: Pass through transformer layers
    DIM layer_idx AS INTEGER
    FOR layer_idx = 0 TO g_config.n_layer - 1
        ' Process the transformer layer
        TransformerLayer model, LayerInput, layer_idx, LayerOutput

        ' If not the last layer, prepare for next iteration
        IF layer_idx < g_config.n_layer - 1 THEN
            ' Copy output to input for next layer
            DIM temp AS Matrix
            temp = LayerInput
            LayerInput = LayerOutput
            LayerOutput = temp
        END IF
    NEXT layer_idx

    ' Step 3: Final Layer Normalization
    DIM FinalNormOutput AS Matrix
    InitMatrix FinalNormOutput, LayerOutput.rows, LayerOutput.cols
    LayerNorm LayerOutput, model.final_ln_gamma, model.final_ln_beta, FinalNormOutput

    ' Step 4: Output Linear Layer (Logits = FinalNormOutput * output_proj)
    MatrixMultiply FinalNormOutput, model.output_proj, Logits

    ' Clean up temporary matrices
    FreeMatrix LayerInput
    FreeMatrix LayerOutput
    FreeMatrix FinalNormOutput
END SUB

SUB SoftmaxVectorFixedPoint(BYREF logits_vector AS Matrix)
    MatrixSoftmaxFixed logits_vector, logits_vector
END SUB

' Function to sample the next token from the logits.
' Input: Logits for the last token in the sequence (a vector of size vocab_size)
' Output: The ID of the sampled token
FUNCTION SampleToken (logits_vector AS Matrix) AS INTEGER
    ' Assumes logits_vector is a matrix with dimensions (1, vocab_size)
    ' This involves applying softmax to the logits and sampling from the resulting probability distribution.

    ' Step 1: Apply Softmax to the logits vector
    ' Convert logits to a probability distribution using our optimized fixed-point Softmax
    SoftmaxVectorFixedPoint logits_vector ' logits_vector will be updated in place with probabilities

    ' Now logits_vector contains properly normalized probabilities as LogQuantized values

    ' Step 2: Sample from the probability distribution
    ' This requires generating a random number and selecting a token based on probabilities.
    ' Need a random number generator and logic to map random number to token ID based on cumulative probabilities.
    ' This can be done with a loop or binary search on cumulative probabilities.
    ' We'll implement two sampling methods:
    ' 1. Greedy sampling (argmax) - always pick the highest probability token
    ' 2. Temperature-based sampling - sample based on the distribution

    ' Define a temperature for sampling (1.0 = use probabilities as-is, <1.0 = more deterministic, >1.0 = more random)
    ' The closer to 0, the more deterministic (similar to argmax)
    CONST TEMPERATURE AS SINGLE = 0.8 ' Adjustable parameter

    ' Flag to toggle between argmax (greedy) and temperature sampling
    CONST USE_TEMPERATURE_SAMPLING AS INTEGER = 1 ' 1 for temperature sampling, 0 for greedy (argmax)

    DIM sampled_token_id AS INTEGER
    DIM c AS INTEGER
    DIM fp_prob AS INTEGER
    DIM fp_temperature AS INTEGER
    DIM fp_adjusted_prob AS INTEGER
    DIM max_prob AS INTEGER
    DIM max_token_id AS INTEGER
    DIM cum_sum AS INTEGER
    DIM fp_rand AS INTEGER

    IF USE_TEMPERATURE_SAMPLING = 0 THEN
        ' Method 1: Greedy sampling (argmax)
        ' Find the token with the highest probability
        max_prob = -2147483647 ' Minimum INT value in fixed-point
        max_token_id = -1

        FOR c = 0 TO g_config.vocab_size - 1
            ' Get probability in fixed-point
            fp_prob = DequantizeLogToFixed(logits_vector.data(0, c))

            IF fp_prob > max_prob THEN
                max_prob = fp_prob
                max_token_id = c
            END IF
        NEXT c

        sampled_token_id = max_token_id ' Argmax sampling
    ELSE
        ' Method 2: Temperature-based sampling
        ' Apply temperature to adjust the probability distribution
        ' Lower temperature makes high probabilities even higher (more deterministic)
        ' Convert temperature to fixed-point
        fp_temperature = FloatToFixed(TEMPERATURE)

        ' Apply temperature by dividing logits by temperature
        FOR c = 0 TO g_config.vocab_size - 1
            fp_prob = DequantizeLogToFixed(logits_vector.data(0, c))
            fp_adjusted_prob = FixedDivide(fp_prob, fp_temperature)
            logits_vector.data(0, c) = FixedToLogQuantized(fp_adjusted_prob).packed_value
        NEXT c

        ' Re-normalize with softmax after temperature adjustment
        SoftmaxVectorFixedPoint logits_vector

        ' Now sample using the adjusted distribution
        ' Simple implementation: Create a cumulative distribution and sample with a random number
        DIM cumulative_probs() AS INTEGER ' Store cumulative probabilities in fixed-point
        REDIM cumulative_probs(0 TO g_config.vocab_size - 1)
        cum_sum = 0

        FOR c = 0 TO g_config.vocab_size - 1
            fp_prob = DequantizeLogToFixed(logits_vector.data(0, c))
            cum_sum = FixedAdd(cum_sum, fp_prob)
            cumulative_probs(c) = cum_sum
        NEXT c

        ' Generate a random number between 0 and cum_sum (the total probability)
        IF cum_sum > 0 THEN
            fp_rand = FloatToFixed(RND * FixedToFloat(cum_sum))
        ELSE
            fp_rand = 0
        END IF

        ' Find the corresponding token
        sampled_token_id = 0
        FOR c = 0 TO g_config.vocab_size - 1
            IF fp_rand <= cumulative_probs(c) THEN
                sampled_token_id = c
                EXIT FOR
            END IF
        NEXT c
    END IF

    FUNCTION = sampled_token_id
END FUNCTION

' Main function for text generation.
' Input: Initial prompt text
' Output: Generated text
SUB GenerateModelText(prompt AS STRING, max_length AS INTEGER)
    DIM token_count AS INTEGER
    DIM output_count AS INTEGER
    DIM input_tokens() AS INTEGER
    DIM output_tokens() AS INTEGER
    DIM i AS INTEGER
    DIM next_token_id AS INTEGER
    DIM start_time AS DOUBLE
    DIM end_time AS DOUBLE
    DIM elapsed AS DOUBLE
    DIM generated_text AS STRING

    PRINT "Generating text starting with: "; prompt

    IF max_length < 1 THEN max_length = 1

    IF GPT2BasicIsLoaded() = 0 THEN
        IF LoadModel("MODEL") = 0 THEN
            PRINT "Failed to load C:\MODEL GPT2-BASIC checkpoint."
            RETURN
        END IF
    END IF

    Encode prompt, input_tokens(), token_count
    IF token_count > 0 THEN
        IF input_tokens(token_count - 1) = 0 THEN token_count = token_count - 1
    END IF
    IF token_count <= 0 THEN
        REDIM input_tokens(0 TO 0)
        input_tokens(0) = 0
        token_count = 1
    END IF
    IF token_count > GPT2BasicContextLength() THEN token_count = GPT2BasicContextLength()

    REDIM output_tokens(0 TO token_count + max_length - 1)
    FOR i = 0 TO token_count - 1
        output_tokens(i) = input_tokens(i)
    NEXT i

    output_count = token_count
    GPT2BasicBeginGeneration token_count

    start_time = TIMER
    FOR i = 1 TO max_length
        next_token_id = GPT2BasicNextToken(output_tokens(), output_count, 0.0, 0.9, 40)
        output_tokens(output_count) = next_token_id
        output_count = output_count + 1
        IF next_token_id = 0 THEN EXIT FOR
    NEXT i
    end_time = TIMER
    elapsed = end_time - start_time
    IF elapsed <= 0.0 THEN elapsed = 0.01

    generated_text = Decode(output_tokens(), output_count)

    PRINT "Checkpoint: "; GPT2BasicProfileName()
    PRINT "Generated "; output_count - token_count; " tokens in "; CompatFormat(elapsed, "0.00"); " seconds."
    PRINT
    PRINT generated_text
    PRINT "Text generation complete."
END SUB

' Main program entry point (example usage)
' SUB main () ' Or equivalent entry point in FreeBASIC
'     GenerateText "The quick brown fox", 50 ' Generate 50 tokens
' END SUB
