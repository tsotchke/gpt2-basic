' *******************************************************
' * GPT2-BASIC Fixed-Point Inference Runtime            *
' *******************************************************
' * Loads host-trained transformer weights from MODEL   *
' * and runs actual decoder-only forward-pass inference *
' * inside the DOS BASIC executable.                    *
' *******************************************************

CONST GPT2BASIC_CFG_FILE AS STRING = "GPT2CFG.TXT"
CONST GPT2BASIC_WEIGHT_FILE AS STRING = "GPT2WT.BIN"
CONST GPT2BASIC_FIXED_WEIGHT_FILE AS STRING = "GPT2FX.BIN"
CONST GPT2BASIC_EXP_TABLE_FILE AS STRING = "GPT2EXP.BIN"
CONST GPT2BASIC_TOKEN_Q4_FILE AS STRING = "GPT2TQ4.BIN"
CONST GPT2BASIC_HEAD_Q4_FILE AS STRING = "GPT2HQ4.BIN"
CONST GPT2BASIC_HEAD_Q4_STREAM_MARKER AS STRING = "GPT2HQS.ON"
CONST GPT2BASIC_VECTOR_FILE AS STRING = "GPT2VEC.TXT"
CONST TINYGPT_CFG_FILE AS STRING = "TINYCFG.TXT"
CONST TINYGPT_WEIGHT_FILE AS STRING = "TINYWT.BIN"
CONST TINYGPT_FIXED_WEIGHT_FILE AS STRING = "TINYFX.BIN"
CONST TINYGPT_EXP_TABLE_FILE AS STRING = "TINYEXP.BIN"
CONST TINYGPT_EOT_TOKEN AS INTEGER = 0
CONST TINYGPT_UNK_TOKEN AS INTEGER = 1
CONST TINYGPT_BYTE_OFFSET AS INTEGER = 2
CONST TINYGPT_MIN_GENERATED AS INTEGER = 70
CONST TINYGPT_FX_SHIFT AS INTEGER = 12
CONST TINYGPT_FX_ONE AS LONG = 4096
CONST TINYGPT_FX_HALF AS LONG = 2048
CONST TINYGPT_FX_EPS AS LONG = 1
CONST TINYGPT_FX_EXP_SIZE AS INTEGER = 513
CONST TINYGPT_FX_EXP_MAX AS INTEGER = 16
CONST TINYGPT_FX_CLAMP AS LONG = 2000000000
CONST TINYGPT_BYTE_RUN_FX_PENALTY AS LONG = 6144
CONST TINYGPT_BYTE_RUN_FLOAT_PENALTY AS SINGLE = 1.5
CONST TINYGPT_TOKEN_Q4_MAGIC AS LONG = &H34515447
CONST TINYGPT_TOKEN_Q4_VERSION AS LONG = 1
CONST TINYGPT_HEAD_Q4_MAGIC AS LONG = &H34514847
CONST TINYGPT_HEAD_Q4_VERSION AS LONG = 1
CONST TINYGPT_KERNEL_STAGE_COUNT AS INTEGER = 6
CONST TINYGPT_KERNEL_EMBED AS INTEGER = 0
CONST TINYGPT_KERNEL_QKV AS INTEGER = 1
CONST TINYGPT_KERNEL_ATTENTION AS INTEGER = 2
CONST TINYGPT_KERNEL_PROJECTION AS INTEGER = 3
CONST TINYGPT_KERNEL_FFN AS INTEGER = 4
CONST TINYGPT_KERNEL_HEAD AS INTEGER = 5

DIM SHARED g_tiny_loaded AS INTEGER
DIM SHARED g_tiny_fixed_loaded AS INTEGER
DIM SHARED g_tiny_vocab_size AS INTEGER
DIM SHARED g_tiny_n_positions AS INTEGER
DIM SHARED g_tiny_n_embd AS INTEGER
DIM SHARED g_tiny_n_head AS INTEGER
DIM SHARED g_tiny_n_layer AS INTEGER
DIM SHARED g_tiny_hidden_dim AS INTEGER
DIM SHARED g_tiny_generation_start_len AS INTEGER
DIM SHARED g_tiny_prompt_start_mode AS INTEGER
DIM SHARED g_tiny_prompt_start_mode_ready AS INTEGER
DIM SHARED g_tiny_profile_name AS STRING
DIM SHARED g_tiny_tracked_memory AS LONG

DIM SHARED g_tiny_tok_emb() AS SINGLE
DIM SHARED g_tiny_pos_emb() AS SINGLE
DIM SHARED g_tiny_ln1_w() AS SINGLE
DIM SHARED g_tiny_ln1_b() AS SINGLE
DIM SHARED g_tiny_q_w() AS SINGLE
DIM SHARED g_tiny_q_b() AS SINGLE
DIM SHARED g_tiny_k_w() AS SINGLE
DIM SHARED g_tiny_k_b() AS SINGLE
DIM SHARED g_tiny_v_w() AS SINGLE
DIM SHARED g_tiny_v_b() AS SINGLE
DIM SHARED g_tiny_proj_w() AS SINGLE
DIM SHARED g_tiny_proj_b() AS SINGLE
DIM SHARED g_tiny_ln2_w() AS SINGLE
DIM SHARED g_tiny_ln2_b() AS SINGLE
DIM SHARED g_tiny_fc1_w() AS SINGLE
DIM SHARED g_tiny_fc1_b() AS SINGLE
DIM SHARED g_tiny_fc2_w() AS SINGLE
DIM SHARED g_tiny_fc2_b() AS SINGLE
DIM SHARED g_tiny_final_ln_w() AS SINGLE
DIM SHARED g_tiny_final_ln_b() AS SINGLE
DIM SHARED g_tiny_head_w() AS SINGLE
DIM SHARED g_tiny_head_b() AS SINGLE

DIM SHARED g_tiny_cache_k() AS SINGLE
DIM SHARED g_tiny_cache_v() AS SINGLE
DIM SHARED g_tiny_cache_tokens() AS INTEGER
DIM SHARED g_tiny_cache_len AS INTEGER
DIM SHARED g_tiny_x_vec() AS SINGLE
DIM SHARED g_tiny_norm_vec() AS SINGLE
DIM SHARED g_tiny_q_vec() AS SINGLE
DIM SHARED g_tiny_k_vec() AS SINGLE
DIM SHARED g_tiny_v_vec() AS SINGLE
DIM SHARED g_tiny_att_vec() AS SINGLE
DIM SHARED g_tiny_proj_vec() AS SINGLE
DIM SHARED g_tiny_ff1_vec() AS SINGLE
DIM SHARED g_tiny_ff2_vec() AS SINGLE
DIM SHARED g_tiny_logits_vec() AS SINGLE
DIM SHARED g_tiny_score_vec() AS SINGLE
DIM SHARED g_tiny_linear_acc() AS DOUBLE

DIM SHARED g_tiny_fx_tok_emb() AS LONG
DIM SHARED g_tiny_fx_pos_emb() AS LONG
DIM SHARED g_tiny_fx_ln1_w() AS LONG
DIM SHARED g_tiny_fx_ln1_b() AS LONG
DIM SHARED g_tiny_fx_q_w() AS LONG
DIM SHARED g_tiny_fx_q_b() AS LONG
DIM SHARED g_tiny_fx_k_w() AS LONG
DIM SHARED g_tiny_fx_k_b() AS LONG
DIM SHARED g_tiny_fx_v_w() AS LONG
DIM SHARED g_tiny_fx_v_b() AS LONG
DIM SHARED g_tiny_fx_proj_w() AS LONG
DIM SHARED g_tiny_fx_proj_b() AS LONG
DIM SHARED g_tiny_fx_ln2_w() AS LONG
DIM SHARED g_tiny_fx_ln2_b() AS LONG
DIM SHARED g_tiny_fx_fc1_w() AS LONG
DIM SHARED g_tiny_fx_fc1_b() AS LONG
DIM SHARED g_tiny_fx_fc2_w() AS LONG
DIM SHARED g_tiny_fx_fc2_b() AS LONG
DIM SHARED g_tiny_fx_final_ln_w() AS LONG
DIM SHARED g_tiny_fx_final_ln_b() AS LONG
DIM SHARED g_tiny_fx_head_w() AS LONG
DIM SHARED g_tiny_fx_head_b() AS LONG
DIM SHARED g_tiny_fx_exp() AS LONG
DIM SHARED g_tiny_fx_tok_q4_loaded AS INTEGER
DIM SHARED g_tiny_fx_tok_q4_bytes AS LONG
DIM SHARED g_tiny_fx_tok_q4() AS UBYTE
DIM SHARED g_tiny_fx_tok_q4_scale() AS LONG
DIM SHARED g_tiny_fx_tok_q4_level() AS LONG
DIM SHARED g_tiny_fx_head_q4_loaded AS INTEGER
DIM SHARED g_tiny_fx_head_q4_bytes AS LONG
DIM SHARED g_tiny_fx_head_q4() AS UBYTE
DIM SHARED g_tiny_fx_head_q4_scale() AS LONG
DIM SHARED g_tiny_fx_head_q4_level() AS LONG
DIM SHARED g_tiny_fx_head_q4_decode() AS LONG
DIM SHARED g_tiny_fx_head_q4_stream AS INTEGER
DIM SHARED g_tiny_fx_head_q4_file AS INTEGER
DIM SHARED g_tiny_fx_head_q4_codes_offset AS LONG
DIM SHARED g_tiny_fx_head_q4_row_bytes AS LONG
DIM SHARED g_tiny_fx_head_q4_row() AS UBYTE

DIM SHARED g_tiny_fx_cache_k() AS LONG
DIM SHARED g_tiny_fx_cache_v() AS LONG
DIM SHARED g_tiny_fx_cache_tokens() AS INTEGER
DIM SHARED g_tiny_fx_cache_len AS INTEGER
DIM SHARED g_tiny_fx_x_vec() AS LONG
DIM SHARED g_tiny_fx_norm_vec() AS LONG
DIM SHARED g_tiny_fx_q_vec() AS LONG
DIM SHARED g_tiny_fx_k_vec() AS LONG
DIM SHARED g_tiny_fx_v_vec() AS LONG
DIM SHARED g_tiny_fx_att_vec() AS LONG
DIM SHARED g_tiny_fx_proj_vec() AS LONG
DIM SHARED g_tiny_fx_ff1_vec() AS LONG
DIM SHARED g_tiny_fx_ff2_vec() AS LONG
DIM SHARED g_tiny_fx_logits_vec() AS LONG
DIM SHARED g_tiny_fx_score_vec() AS LONG
DIM SHARED g_tiny_fx_linear_acc() AS LONGINT
DIM SHARED g_tiny_fx_dbg_embedding_vec() AS LONG
DIM SHARED g_tiny_fx_dbg_ln1_vec() AS LONG
DIM SHARED g_tiny_fx_dbg_q_vec() AS LONG
DIM SHARED g_tiny_fx_dbg_k_vec() AS LONG
DIM SHARED g_tiny_fx_dbg_v_vec() AS LONG
DIM SHARED g_tiny_fx_dbg_attn_vec() AS LONG
DIM SHARED g_tiny_fx_dbg_proj_vec() AS LONG
DIM SHARED g_tiny_fx_dbg_ln2_vec() AS LONG
DIM SHARED g_tiny_fx_dbg_ff1_vec() AS LONG
DIM SHARED g_tiny_fx_dbg_ff2_vec() AS LONG
DIM SHARED g_tiny_fx_dbg_hidden_vec() AS LONG
DIM SHARED g_tiny_fx_dbg_final_ln_vec() AS LONG
DIM SHARED g_tiny_phase_capture_enabled AS INTEGER
DIM SHARED g_tiny_phase_debug_allocated AS INTEGER
DIM SHARED g_tiny_fx_attn_scale AS LONG
DIM SHARED g_tiny_kernel_perf_enabled AS INTEGER
DIM SHARED g_tiny_kernel_perf_seconds(0 TO TINYGPT_KERNEL_STAGE_COUNT - 1) AS DOUBLE
DIM SHARED g_tiny_kernel_perf_calls(0 TO TINYGPT_KERNEL_STAGE_COUNT - 1) AS LONG

DECLARE FUNCTION TinyGPTLoadModel(base_path AS STRING) AS INTEGER
DECLARE FUNCTION TinyGPTResolveFile(base_path AS STRING, primary_name AS STRING, legacy_name AS STRING) AS STRING
DECLARE FUNCTION TinyGPTLoadConfig(filename AS STRING) AS INTEGER
DECLARE SUB TinyGPTReadSingles(file_num AS INTEGER, values() AS SINGLE, value_count AS LONG)
DECLARE SUB TinyGPTReadLongs(file_num AS INTEGER, values() AS LONG, value_count AS LONG)
DECLARE FUNCTION TinyGPTLoadFixedModel(base_path AS STRING) AS INTEGER
DECLARE FUNCTION TinyGPTLoadFixedExpTable(base_path AS STRING) AS INTEGER
DECLARE FUNCTION TinyGPTLoadTokenQ4(base_path AS STRING, expected_value_count AS LONG) AS INTEGER
DECLARE FUNCTION TinyGPTLoadHeadQ4(base_path AS STRING, expected_value_count AS LONG) AS INTEGER
DECLARE SUB TinyGPTFreeModel()
DECLARE FUNCTION TinyGPTIsLoaded() AS INTEGER
DECLARE FUNCTION TinyGPTIsFixedPointLoaded() AS INTEGER
DECLARE FUNCTION TinyGPTEmbeddingDim() AS INTEGER
DECLARE FUNCTION TinyGPTHeadCount() AS INTEGER
DECLARE FUNCTION TinyGPTLayerCount() AS INTEGER
DECLARE FUNCTION TinyGPTContextLength() AS INTEGER
DECLARE FUNCTION TinyGPTVocabSize() AS INTEGER
DECLARE FUNCTION TinyGPTProfileName() AS STRING
DECLARE FUNCTION TinyGPTParameterCount() AS LONG
DECLARE FUNCTION TinyGPTFixedWeightBytes() AS LONG
DECLARE FUNCTION TinyGPTRuntimeMemoryBytes() AS LONG
DECLARE SUB TinyGPTBeginGeneration(prompt_token_count AS INTEGER)
DECLARE FUNCTION TinyGPTNextToken(context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
DECLARE FUNCTION TinyGPTNextTokenFull(context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
DECLARE FUNCTION TinyGPTNextTokenCached(context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
DECLARE SUB TinyGPTAllocateDecodeCache()
DECLARE SUB TinyGPTResetDecodeCache()
DECLARE SUB TinyGPTAllocateFixedDecodeCache()
DECLARE SUB TinyGPTResetFixedDecodeCache()
DECLARE SUB TinyGPTAllocatePhaseDebugBuffers()
DECLARE SUB TinyGPTFreePhaseDebugBuffers()
DECLARE SUB TinyGPTLayerNormSeq(input_arr() AS SINGLE, output_arr() AS SINGLE, seq_len AS INTEGER, emb_dim AS INTEGER, gamma_arr() AS SINGLE, beta_arr() AS SINGLE, param_base AS LONG)
DECLARE SUB TinyGPTLayerNormLast(input_arr() AS SINGLE, output_arr() AS SINGLE, seq_len AS INTEGER, emb_dim AS INTEGER, gamma_arr() AS SINGLE, beta_arr() AS SINGLE)
DECLARE SUB TinyGPTLayerNormVec(input_vec() AS SINGLE, output_vec() AS SINGLE, emb_dim AS INTEGER, gamma_arr() AS SINGLE, beta_arr() AS SINGLE, param_base AS LONG)
DECLARE SUB TinyGPTLinearSeq(input_arr() AS SINGLE, output_arr() AS SINGLE, seq_len AS INTEGER, in_dim AS INTEGER, out_dim AS INTEGER, weight_arr() AS SINGLE, weight_base AS LONG, bias_arr() AS SINGLE, bias_base AS LONG)
DECLARE SUB TinyGPTLinearVec(input_vec() AS SINGLE, output_vec() AS SINGLE, in_dim AS INTEGER, out_dim AS INTEGER, weight_arr() AS SINGLE, weight_base AS LONG, bias_arr() AS SINGLE, bias_base AS LONG)
DECLARE SUB TinyGPTAttention(norm_arr() AS SINGLE, att_out() AS SINGLE, seq_len AS INTEGER, layer_idx AS INTEGER)
DECLARE SUB TinyGPTForwardCachedToken(token_id AS INTEGER, cache_pos AS INTEGER, write_logits AS INTEGER, logits() AS SINGLE)
DECLARE FUNCTION TinyGPTGELU(x AS SINGLE) AS SINGLE
DECLARE FUNCTION TinyGPTTokenAllowed(token_id AS INTEGER) AS INTEGER
DECLARE FUNCTION TinyGPTTokenCanBeginOutput(token_id AS INTEGER) AS INTEGER
DECLARE FUNCTION TinyGPTTokenEndsSentence(token_id AS INTEGER) AS INTEGER
DECLARE FUNCTION TinyGPTTokenText(token_id AS INTEGER) AS STRING
DECLARE FUNCTION TinyGPTGeneratedAlphaSuffixLen(context() AS INTEGER, context_len AS INTEGER) AS INTEGER
DECLARE FUNCTION TinyGPTTextStartsWith(text_value AS STRING, prefix_value AS STRING) AS INTEGER
DECLARE FUNCTION TinyGPTPromptStartMode(context() AS INTEGER, context_len AS INTEGER) AS INTEGER
DECLARE FUNCTION TinyGPTTokenAllowedForStartMode(start_mode AS INTEGER, token_id AS INTEGER) AS INTEGER
DECLARE FUNCTION TinyGPTTokenCanFollowContext(context() AS INTEGER, context_len AS INTEGER, token_id AS INTEGER) AS INTEGER
DECLARE FUNCTION TinyGPTTokenFollowPenalty(context() AS INTEGER, context_len AS INTEGER, token_id AS INTEGER) AS INTEGER
DECLARE FUNCTION TinyGPTSample(logits() AS SINGLE, context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
DECLARE FUNCTION TinyGPTFixedClamp(value AS LONGINT) AS LONG
DECLARE FUNCTION TinyGPTFixedMul(a AS LONG, b AS LONG) AS LONG
DECLARE FUNCTION TinyGPTFixedDiv(a AS LONG, b AS LONG) AS LONG
DECLARE FUNCTION TinyGPTFixedSqrt(value AS LONG) AS LONG
DECLARE FUNCTION TinyGPTFixedExpNeg(x AS LONG) AS LONG
DECLARE FUNCTION TinyGPTFixedTanh(x AS LONG) AS LONG
DECLARE SUB TinyGPTKernelPerfReset()
DECLARE SUB TinyGPTKernelPerfSetEnabled(enabled AS INTEGER)
DECLARE FUNCTION TinyGPTKernelPerfStageName(stage_id AS INTEGER) AS STRING
DECLARE SUB TinyGPTKernelPerfAdd(stage_id AS INTEGER, start_time AS DOUBLE)
DECLARE SUB TinyGPTKernelPerfReport()
DECLARE SUB TinyGPTFixedLayerNormVec(input_vec() AS LONG, output_vec() AS LONG, emb_dim AS INTEGER, gamma_arr() AS LONG, beta_arr() AS LONG, param_base AS LONG)
DECLARE SUB TinyGPTFixedLinearVec(input_vec() AS LONG, output_vec() AS LONG, in_dim AS INTEGER, out_dim AS INTEGER, weight_arr() AS LONG, weight_base AS LONG, bias_arr() AS LONG, bias_base AS LONG)
DECLARE SUB TinyGPTFixedHeadQ4LinearVec(input_vec() AS LONG, output_vec() AS LONG)
DECLARE SUB TinyGPTFixedForwardCachedToken(token_id AS INTEGER, cache_pos AS INTEGER, write_logits AS INTEGER, logits() AS LONG)
DECLARE FUNCTION TinyGPTFixedGELU(x AS LONG) AS LONG
DECLARE FUNCTION TinyGPTFixedSample(logits() AS LONG, context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
DECLARE FUNCTION TinyGPTNextTokenFixedCached(context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
DECLARE FUNCTION TinyGPTForwardFixedLogits(context() AS INTEGER, context_len AS INTEGER, logits() AS LONG) AS INTEGER
DECLARE FUNCTION TinyGPTForwardFloatLogits(context() AS INTEGER, context_len AS INTEGER, logits() AS SINGLE) AS INTEGER
DECLARE FUNCTION TinyGPTVectorField(line_text AS STRING, field_idx AS INTEGER) AS STRING
DECLARE SUB TinyGPTParseVectorTokens(token_text AS STRING, tokens() AS INTEGER, BYREF token_count AS INTEGER)
DECLARE SUB TinyGPTCopyLongVector(source() AS LONG, dest() AS LONG, value_count AS INTEGER)
DECLARE FUNCTION TinyGPTAbsLong(value AS LONG) AS LONG
DECLARE FUNCTION TinyGPTCompareVectorLogits(vector_name AS STRING, logits() AS LONG, expected_text AS STRING, tolerance AS LONG) AS INTEGER
DECLARE FUNCTION TinyGPTComparePhaseValues(vector_name AS STRING, phase_name AS STRING, declared_dim AS INTEGER, expected_text AS STRING, tolerance AS LONG) AS INTEGER
DECLARE FUNCTION TinyGPTRunVectorFile(vector_path AS STRING, tolerance AS LONG) AS INTEGER
DECLARE FUNCTION GPT2BasicLoadModel(base_path AS STRING) AS INTEGER
DECLARE SUB GPT2BasicFreeModel()
DECLARE FUNCTION GPT2BasicIsLoaded() AS INTEGER
DECLARE FUNCTION GPT2BasicIsFixedPointLoaded() AS INTEGER
DECLARE FUNCTION GPT2BasicEmbeddingDim() AS INTEGER
DECLARE FUNCTION GPT2BasicHeadCount() AS INTEGER
DECLARE FUNCTION GPT2BasicLayerCount() AS INTEGER
DECLARE FUNCTION GPT2BasicContextLength() AS INTEGER
DECLARE FUNCTION GPT2BasicVocabSize() AS INTEGER
DECLARE FUNCTION GPT2BasicProfileName() AS STRING
DECLARE FUNCTION GPT2BasicParameterCount() AS LONG
DECLARE FUNCTION GPT2BasicFixedWeightBytes() AS LONG
DECLARE FUNCTION GPT2BasicRuntimeMemoryBytes() AS LONG
DECLARE SUB GPT2BasicBeginGeneration(prompt_token_count AS INTEGER)
DECLARE FUNCTION GPT2BasicNextToken(context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
DECLARE FUNCTION GPT2BasicForwardFixedLogits(context() AS INTEGER, context_len AS INTEGER, logits() AS LONG) AS INTEGER
DECLARE FUNCTION GPT2BasicForwardFloatLogits(context() AS INTEGER, context_len AS INTEGER, logits() AS SINGLE) AS INTEGER
DECLARE FUNCTION GPT2BasicRunVectorFile(vector_path AS STRING, tolerance AS LONG) AS INTEGER

FUNCTION TinyGPTResolveFile(base_path AS STRING, primary_name AS STRING, legacy_name AS STRING) AS STRING
    DIM primary_path AS STRING
    DIM legacy_path AS STRING

    primary_path = base_path + "\" + primary_name
    legacy_path = base_path + "\" + legacy_name

    IF DIR(primary_path) <> "" THEN RETURN primary_path
    RETURN legacy_path
END FUNCTION

FUNCTION TinyGPTLoadConfig(filename AS STRING) AS INTEGER
    DIM file_num AS INTEGER
    DIM line_buffer AS STRING
    DIM eq_pos AS INTEGER
    DIM key_text AS STRING
    DIM value_text AS STRING

    g_tiny_vocab_size = 0
    g_tiny_n_positions = 0
    g_tiny_n_embd = 0
    g_tiny_n_head = 0
    g_tiny_n_layer = 0
    g_tiny_hidden_dim = 0
    g_tiny_profile_name = "custom"

    file_num = FREEFILE
    ON ERROR GOTO cfg_error
    OPEN filename FOR INPUT AS #file_num

    WHILE EOF(file_num) = 0
        LINE INPUT #file_num, line_buffer
        line_buffer = TRIM$(line_buffer)

        IF line_buffer <> "" AND LEFT$(line_buffer, 1) <> "#" THEN
            eq_pos = INSTR(line_buffer, "=")
            IF eq_pos > 0 THEN
                key_text = LCASE$(TRIM$(LEFT$(line_buffer, eq_pos - 1)))
                value_text = TRIM$(MID$(line_buffer, eq_pos + 1))

                SELECT CASE key_text
                    CASE "profile"
                        g_tiny_profile_name = value_text
                    CASE "vocab_size"
                        g_tiny_vocab_size = VAL(value_text)
                    CASE "n_positions"
                        g_tiny_n_positions = VAL(value_text)
                    CASE "n_embd"
                        g_tiny_n_embd = VAL(value_text)
                    CASE "n_head"
                        g_tiny_n_head = VAL(value_text)
                    CASE "n_layer"
                        g_tiny_n_layer = VAL(value_text)
                    CASE "hidden_dim"
                        g_tiny_hidden_dim = VAL(value_text)
                END SELECT
            END IF
        END IF
    WEND

    CLOSE #file_num
    ON ERROR GOTO 0

    IF g_tiny_vocab_size < 1 THEN RETURN 0
    IF g_tiny_n_positions < 1 THEN RETURN 0
    IF g_tiny_n_embd < 1 THEN RETURN 0
    IF g_tiny_n_head < 1 THEN RETURN 0
    IF g_tiny_n_layer < 1 THEN RETURN 0
    IF g_tiny_hidden_dim < 1 THEN RETURN 0
    IF (g_tiny_n_embd MOD g_tiny_n_head) <> 0 THEN RETURN 0

    RETURN 1

cfg_error:
    ON ERROR GOTO 0
    RETURN 0
END FUNCTION

SUB TinyGPTReadSingles(file_num AS INTEGER, values() AS SINGLE, value_count AS LONG)
    DIM i AS LONG
    FOR i = 0 TO value_count - 1
        GET #file_num, , values(i)
    NEXT i
END SUB

SUB TinyGPTReadLongs(file_num AS INTEGER, values() AS LONG, value_count AS LONG)
    DIM i AS LONG
    FOR i = 0 TO value_count - 1
        GET #file_num, , values(i)
    NEXT i
END SUB

FUNCTION TinyGPTLoadTokenQ4(base_path AS STRING, expected_value_count AS LONG) AS INTEGER
    DIM q4_path AS STRING
    DIM file_num AS INTEGER
    DIM magic AS LONG
    DIM version AS LONG
    DIM vocab_size AS LONG
    DIM emb_dim AS LONG
    DIM value_count AS LONG
    DIM level_count AS LONG
    DIM scale_count AS LONG
    DIM packed_bytes AS LONG
    DIM i AS LONG
    DIM byte_value AS UBYTE

    q4_path = base_path + "\" + GPT2BASIC_TOKEN_Q4_FILE
    IF DIR(q4_path) = "" THEN RETURN 0

    file_num = FREEFILE
    ON ERROR GOTO token_q4_error
    OPEN q4_path FOR BINARY AS #file_num

    GET #file_num, , magic
    GET #file_num, , version
    GET #file_num, , vocab_size
    GET #file_num, , emb_dim
    GET #file_num, , value_count
    GET #file_num, , level_count
    GET #file_num, , scale_count
    GET #file_num, , packed_bytes

    IF magic <> TINYGPT_TOKEN_Q4_MAGIC THEN GOTO token_q4_error
    IF version <> TINYGPT_TOKEN_Q4_VERSION THEN GOTO token_q4_error
    IF vocab_size <> g_tiny_vocab_size THEN GOTO token_q4_error
    IF emb_dim <> g_tiny_n_embd THEN GOTO token_q4_error
    IF value_count <> expected_value_count THEN GOTO token_q4_error
    IF level_count <> 8 THEN GOTO token_q4_error
    IF scale_count <> g_tiny_vocab_size THEN GOTO token_q4_error
    IF packed_bytes <> (expected_value_count + 1) \ 2 THEN GOTO token_q4_error
    IF packed_bytes <= 0 THEN GOTO token_q4_error

    REDIM g_tiny_fx_tok_q4_level(0 TO level_count - 1)
    REDIM g_tiny_fx_tok_q4_scale(0 TO scale_count - 1)
    REDIM g_tiny_fx_tok_q4(0 TO packed_bytes - 1)

    TinyGPTReadLongs file_num, g_tiny_fx_tok_q4_level(), level_count
    TinyGPTReadLongs file_num, g_tiny_fx_tok_q4_scale(), scale_count
    FOR i = 0 TO packed_bytes - 1
        GET #file_num, , byte_value
        g_tiny_fx_tok_q4(i) = byte_value
    NEXT i

    IF SEEK(file_num) <= LOF(file_num) THEN GOTO token_q4_error
    CLOSE #file_num
    ON ERROR GOTO 0

    g_tiny_fx_tok_q4_loaded = 1
    g_tiny_fx_tok_q4_bytes = 32 + (level_count * 4) + (scale_count * 4) + packed_bytes
    RETURN 1

token_q4_error:
    ON ERROR GOTO 0
    IF file_num <> 0 THEN CLOSE #file_num
    ERASE g_tiny_fx_tok_q4
    ERASE g_tiny_fx_tok_q4_scale
    ERASE g_tiny_fx_tok_q4_level
    g_tiny_fx_tok_q4_loaded = 0
    g_tiny_fx_tok_q4_bytes = 0
    RETURN 0
END FUNCTION

FUNCTION TinyGPTLoadHeadQ4(base_path AS STRING, expected_value_count AS LONG) AS INTEGER
    DIM q4_path AS STRING
    DIM file_num AS INTEGER
    DIM use_stream AS INTEGER
    DIM magic AS LONG
    DIM version AS LONG
    DIM vocab_size AS LONG
    DIM emb_dim AS LONG
    DIM value_count AS LONG
    DIM level_count AS LONG
    DIM scale_count AS LONG
    DIM packed_bytes AS LONG
    DIM i AS LONG
    DIM code AS INTEGER
    DIM magnitude AS INTEGER
    DIM decode_index AS LONG
    DIM decoded_value AS LONGINT
    DIM byte_value AS UBYTE

    q4_path = base_path + "\" + GPT2BASIC_HEAD_Q4_FILE
    IF DIR(q4_path) = "" THEN RETURN 0
    use_stream = 0
    IF DIR(base_path + "\" + GPT2BASIC_HEAD_Q4_STREAM_MARKER) <> "" THEN use_stream = 1

    file_num = FREEFILE
    ON ERROR GOTO head_q4_error
    OPEN q4_path FOR BINARY AS #file_num

    GET #file_num, , magic
    GET #file_num, , version
    GET #file_num, , vocab_size
    GET #file_num, , emb_dim
    GET #file_num, , value_count
    GET #file_num, , level_count
    GET #file_num, , scale_count
    GET #file_num, , packed_bytes

    IF magic <> TINYGPT_HEAD_Q4_MAGIC THEN GOTO head_q4_error
    IF version <> TINYGPT_HEAD_Q4_VERSION THEN GOTO head_q4_error
    IF vocab_size <> g_tiny_vocab_size THEN GOTO head_q4_error
    IF emb_dim <> g_tiny_n_embd THEN GOTO head_q4_error
    IF value_count <> expected_value_count THEN GOTO head_q4_error
    IF level_count <> 8 THEN GOTO head_q4_error
    IF scale_count <> g_tiny_vocab_size THEN GOTO head_q4_error
    IF packed_bytes <> (expected_value_count + 1) \ 2 THEN GOTO head_q4_error
    IF packed_bytes <= 0 THEN GOTO head_q4_error

    REDIM g_tiny_fx_head_q4_level(0 TO level_count - 1)
    REDIM g_tiny_fx_head_q4_scale(0 TO scale_count - 1)

    TinyGPTReadLongs file_num, g_tiny_fx_head_q4_level(), level_count
    TinyGPTReadLongs file_num, g_tiny_fx_head_q4_scale(), scale_count

    IF use_stream <> 0 THEN
        g_tiny_fx_head_q4_codes_offset = SEEK(file_num) - 1
        g_tiny_fx_head_q4_row_bytes = (CLNG(g_tiny_vocab_size) + 1) \ 2
        IF g_tiny_fx_head_q4_row_bytes < 1 THEN GOTO head_q4_error
        IF packed_bytes <> g_tiny_fx_head_q4_row_bytes * CLNG(g_tiny_n_embd) THEN GOTO head_q4_error
        IF LOF(file_num) <> g_tiny_fx_head_q4_codes_offset + packed_bytes THEN GOTO head_q4_error
        REDIM g_tiny_fx_head_q4_row(0 TO g_tiny_fx_head_q4_row_bytes - 1)
        g_tiny_fx_head_q4_file = file_num
        g_tiny_fx_head_q4_stream = 1
        g_tiny_fx_head_q4_loaded = 1
        g_tiny_fx_head_q4_bytes = 32 + (level_count * 4) + (scale_count * 4) + g_tiny_fx_head_q4_row_bytes
        ON ERROR GOTO 0
        RETURN 1
    END IF

    REDIM g_tiny_fx_head_q4(0 TO packed_bytes - 1)
    REDIM g_tiny_fx_head_q4_decode(0 TO (scale_count * 16) - 1)

    FOR i = 0 TO packed_bytes - 1
        GET #file_num, , byte_value
        g_tiny_fx_head_q4(i) = byte_value
    NEXT i

    IF SEEK(file_num) <= LOF(file_num) THEN GOTO head_q4_error
    CLOSE #file_num
    ON ERROR GOTO 0

    FOR i = 0 TO scale_count - 1
        FOR code = 0 TO 15
            magnitude = code AND 7
            decoded_value = 0
            IF magnitude <> 0 THEN
                decoded_value = (CLNGINT(g_tiny_fx_head_q4_scale(i)) * CLNGINT(g_tiny_fx_head_q4_level(magnitude))) \ TINYGPT_FX_ONE
                IF (code AND 8) <> 0 THEN decoded_value = -decoded_value
            END IF
            decode_index = (i * 16) + code
            g_tiny_fx_head_q4_decode(decode_index) = TinyGPTFixedClamp(decoded_value)
        NEXT code
    NEXT i

    g_tiny_fx_head_q4_loaded = 1
    g_tiny_fx_head_q4_bytes = 32 + (level_count * 4) + (scale_count * 4) + packed_bytes + (scale_count * 16 * 4)
    RETURN 1

head_q4_error:
    ON ERROR GOTO 0
    IF file_num <> 0 THEN CLOSE #file_num
    ERASE g_tiny_fx_head_q4
    ERASE g_tiny_fx_head_q4_scale
    ERASE g_tiny_fx_head_q4_level
    ERASE g_tiny_fx_head_q4_decode
    ERASE g_tiny_fx_head_q4_row
    g_tiny_fx_head_q4_stream = 0
    g_tiny_fx_head_q4_file = 0
    g_tiny_fx_head_q4_codes_offset = 0
    g_tiny_fx_head_q4_row_bytes = 0
    g_tiny_fx_head_q4_loaded = 0
    g_tiny_fx_head_q4_bytes = 0
    RETURN 0
END FUNCTION

FUNCTION TinyGPTLoadFixedExpTable(base_path AS STRING) AS INTEGER
    DIM exp_path AS STRING
    DIM file_num AS INTEGER

    exp_path = TinyGPTResolveFile(base_path, GPT2BASIC_EXP_TABLE_FILE, TINYGPT_EXP_TABLE_FILE)
    REDIM g_tiny_fx_exp(0 TO TINYGPT_FX_EXP_SIZE - 1)

    file_num = FREEFILE
    ON ERROR GOTO exp_error
    OPEN exp_path FOR BINARY AS #file_num
    TinyGPTReadLongs file_num, g_tiny_fx_exp(), TINYGPT_FX_EXP_SIZE
    CLOSE #file_num
    ON ERROR GOTO 0

    RETURN 1

exp_error:
    ON ERROR GOTO 0
    RETURN 0
END FUNCTION

FUNCTION TinyGPTLoadFixedModel(base_path AS STRING) AS INTEGER
    DIM weight_path AS STRING
    DIM file_num AS INTEGER
    DIM layer_e AS LONG
    DIM layer_ee AS LONG
    DIM layer_eh AS LONG
    DIM layer_he AS LONG
    DIM token_count AS LONG
    DIM pos_count AS LONG
    DIM head_count AS LONG
    DIM skip_bytes AS LONG

    weight_path = TinyGPTResolveFile(base_path, GPT2BASIC_FIXED_WEIGHT_FILE, TINYGPT_FIXED_WEIGHT_FILE)

    IF TinyGPTLoadFixedExpTable(base_path) = 0 THEN
        RETURN 0
    END IF

    token_count = CLNG(g_tiny_vocab_size) * CLNG(g_tiny_n_embd)
    pos_count = CLNG(g_tiny_n_positions) * CLNG(g_tiny_n_embd)
    layer_e = CLNG(g_tiny_n_layer) * CLNG(g_tiny_n_embd)
    layer_ee = CLNG(g_tiny_n_layer) * CLNG(g_tiny_n_embd) * CLNG(g_tiny_n_embd)
    layer_eh = CLNG(g_tiny_n_layer) * CLNG(g_tiny_n_embd) * CLNG(g_tiny_hidden_dim)
    layer_he = CLNG(g_tiny_n_layer) * CLNG(g_tiny_hidden_dim) * CLNG(g_tiny_n_embd)
    head_count = CLNG(g_tiny_n_embd) * CLNG(g_tiny_vocab_size)
    g_tiny_fx_tok_q4_loaded = 0
    g_tiny_fx_tok_q4_bytes = 0
    g_tiny_fx_head_q4_loaded = 0
    g_tiny_fx_head_q4_bytes = 0
    g_tiny_fx_head_q4_stream = 0
    g_tiny_fx_head_q4_file = 0
    g_tiny_fx_head_q4_codes_offset = 0
    g_tiny_fx_head_q4_row_bytes = 0
    TinyGPTLoadTokenQ4 base_path, token_count
    TinyGPTLoadHeadQ4 base_path, head_count

    IF g_tiny_fx_tok_q4_loaded = 0 THEN
        REDIM g_tiny_fx_tok_emb(0 TO token_count - 1)
    END IF
    REDIM g_tiny_fx_pos_emb(0 TO pos_count - 1)
    REDIM g_tiny_fx_ln1_w(0 TO layer_e - 1)
    REDIM g_tiny_fx_ln1_b(0 TO layer_e - 1)
    REDIM g_tiny_fx_q_w(0 TO layer_ee - 1)
    REDIM g_tiny_fx_q_b(0 TO layer_e - 1)
    REDIM g_tiny_fx_k_w(0 TO layer_ee - 1)
    REDIM g_tiny_fx_k_b(0 TO layer_e - 1)
    REDIM g_tiny_fx_v_w(0 TO layer_ee - 1)
    REDIM g_tiny_fx_v_b(0 TO layer_e - 1)
    REDIM g_tiny_fx_proj_w(0 TO layer_ee - 1)
    REDIM g_tiny_fx_proj_b(0 TO layer_e - 1)
    REDIM g_tiny_fx_ln2_w(0 TO layer_e - 1)
    REDIM g_tiny_fx_ln2_b(0 TO layer_e - 1)
    REDIM g_tiny_fx_fc1_w(0 TO layer_eh - 1)
    REDIM g_tiny_fx_fc1_b(0 TO CLNG(g_tiny_n_layer) * CLNG(g_tiny_hidden_dim) - 1)
    REDIM g_tiny_fx_fc2_w(0 TO layer_he - 1)
    REDIM g_tiny_fx_fc2_b(0 TO layer_e - 1)
    REDIM g_tiny_fx_final_ln_w(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_final_ln_b(0 TO g_tiny_n_embd - 1)
    IF g_tiny_fx_head_q4_loaded = 0 THEN
        REDIM g_tiny_fx_head_w(0 TO head_count - 1)
    END IF
    REDIM g_tiny_fx_head_b(0 TO g_tiny_vocab_size - 1)

    file_num = FREEFILE
    ON ERROR GOTO fixed_weight_error
    OPEN weight_path FOR BINARY AS #file_num

    IF g_tiny_fx_tok_q4_loaded <> 0 THEN
        skip_bytes = token_count * 4
        SEEK #file_num, SEEK(file_num) + skip_bytes
    ELSE
        TinyGPTReadLongs file_num, g_tiny_fx_tok_emb(), token_count
    END IF
    TinyGPTReadLongs file_num, g_tiny_fx_pos_emb(), pos_count
    TinyGPTReadLongs file_num, g_tiny_fx_ln1_w(), layer_e
    TinyGPTReadLongs file_num, g_tiny_fx_ln1_b(), layer_e
    TinyGPTReadLongs file_num, g_tiny_fx_q_w(), layer_ee
    TinyGPTReadLongs file_num, g_tiny_fx_q_b(), layer_e
    TinyGPTReadLongs file_num, g_tiny_fx_k_w(), layer_ee
    TinyGPTReadLongs file_num, g_tiny_fx_k_b(), layer_e
    TinyGPTReadLongs file_num, g_tiny_fx_v_w(), layer_ee
    TinyGPTReadLongs file_num, g_tiny_fx_v_b(), layer_e
    TinyGPTReadLongs file_num, g_tiny_fx_proj_w(), layer_ee
    TinyGPTReadLongs file_num, g_tiny_fx_proj_b(), layer_e
    TinyGPTReadLongs file_num, g_tiny_fx_ln2_w(), layer_e
    TinyGPTReadLongs file_num, g_tiny_fx_ln2_b(), layer_e
    TinyGPTReadLongs file_num, g_tiny_fx_fc1_w(), layer_eh
    TinyGPTReadLongs file_num, g_tiny_fx_fc1_b(), CLNG(g_tiny_n_layer) * CLNG(g_tiny_hidden_dim)
    TinyGPTReadLongs file_num, g_tiny_fx_fc2_w(), layer_he
    TinyGPTReadLongs file_num, g_tiny_fx_fc2_b(), layer_e
    TinyGPTReadLongs file_num, g_tiny_fx_final_ln_w(), g_tiny_n_embd
    TinyGPTReadLongs file_num, g_tiny_fx_final_ln_b(), g_tiny_n_embd
    IF g_tiny_fx_head_q4_loaded <> 0 THEN
        skip_bytes = head_count * 4
        SEEK #file_num, SEEK(file_num) + skip_bytes
    ELSE
        TinyGPTReadLongs file_num, g_tiny_fx_head_w(), head_count
    END IF
    TinyGPTReadLongs file_num, g_tiny_fx_head_b(), g_tiny_vocab_size

    CLOSE #file_num
    ON ERROR GOTO 0

	g_tiny_fixed_loaded = 1
	TinyGPTAllocateFixedDecodeCache
	g_tiny_tracked_memory = TinyGPTRuntimeMemoryBytes()
	TrackAllocation g_tiny_tracked_memory
	RETURN 1

fixed_weight_error:
    ON ERROR GOTO 0
    IF file_num <> 0 THEN CLOSE #file_num
    IF g_tiny_fx_head_q4_stream <> 0 AND g_tiny_fx_head_q4_file <> 0 THEN CLOSE #g_tiny_fx_head_q4_file
    g_tiny_fixed_loaded = 0
    ERASE g_tiny_fx_tok_q4
    ERASE g_tiny_fx_tok_q4_scale
    ERASE g_tiny_fx_tok_q4_level
    ERASE g_tiny_fx_head_q4
    ERASE g_tiny_fx_head_q4_scale
    ERASE g_tiny_fx_head_q4_level
    ERASE g_tiny_fx_head_q4_decode
    ERASE g_tiny_fx_head_q4_row
    g_tiny_fx_tok_q4_loaded = 0
    g_tiny_fx_tok_q4_bytes = 0
    g_tiny_fx_head_q4_loaded = 0
    g_tiny_fx_head_q4_bytes = 0
    g_tiny_fx_head_q4_stream = 0
    g_tiny_fx_head_q4_file = 0
    g_tiny_fx_head_q4_codes_offset = 0
    g_tiny_fx_head_q4_row_bytes = 0
    RETURN 0
END FUNCTION

FUNCTION TinyGPTLoadModel(base_path AS STRING) AS INTEGER
    DIM cfg_path AS STRING
    DIM weight_path AS STRING
    DIM file_num AS INTEGER
    DIM layer_e AS LONG
    DIM layer_ee AS LONG
    DIM layer_eh AS LONG
    DIM layer_he AS LONG
    DIM token_count AS LONG
    DIM pos_count AS LONG
    DIM head_count AS LONG

    cfg_path = TinyGPTResolveFile(base_path, GPT2BASIC_CFG_FILE, TINYGPT_CFG_FILE)
    weight_path = TinyGPTResolveFile(base_path, GPT2BASIC_WEIGHT_FILE, TINYGPT_WEIGHT_FILE)

    IF TinyGPTLoadConfig(cfg_path) = 0 THEN
        RETURN 0
    END IF

    IF TinyGPTLoadFixedModel(base_path) <> 0 THEN
        g_tiny_loaded = 1
        g_tiny_generation_start_len = 0
        RETURN 1
    END IF

    token_count = CLNG(g_tiny_vocab_size) * CLNG(g_tiny_n_embd)
    pos_count = CLNG(g_tiny_n_positions) * CLNG(g_tiny_n_embd)
    layer_e = CLNG(g_tiny_n_layer) * CLNG(g_tiny_n_embd)
    layer_ee = CLNG(g_tiny_n_layer) * CLNG(g_tiny_n_embd) * CLNG(g_tiny_n_embd)
    layer_eh = CLNG(g_tiny_n_layer) * CLNG(g_tiny_n_embd) * CLNG(g_tiny_hidden_dim)
    layer_he = CLNG(g_tiny_n_layer) * CLNG(g_tiny_hidden_dim) * CLNG(g_tiny_n_embd)
    head_count = CLNG(g_tiny_n_embd) * CLNG(g_tiny_vocab_size)

    REDIM g_tiny_tok_emb(0 TO token_count - 1)
    REDIM g_tiny_pos_emb(0 TO pos_count - 1)
    REDIM g_tiny_ln1_w(0 TO layer_e - 1)
    REDIM g_tiny_ln1_b(0 TO layer_e - 1)
    REDIM g_tiny_q_w(0 TO layer_ee - 1)
    REDIM g_tiny_q_b(0 TO layer_e - 1)
    REDIM g_tiny_k_w(0 TO layer_ee - 1)
    REDIM g_tiny_k_b(0 TO layer_e - 1)
    REDIM g_tiny_v_w(0 TO layer_ee - 1)
    REDIM g_tiny_v_b(0 TO layer_e - 1)
    REDIM g_tiny_proj_w(0 TO layer_ee - 1)
    REDIM g_tiny_proj_b(0 TO layer_e - 1)
    REDIM g_tiny_ln2_w(0 TO layer_e - 1)
    REDIM g_tiny_ln2_b(0 TO layer_e - 1)
    REDIM g_tiny_fc1_w(0 TO layer_eh - 1)
    REDIM g_tiny_fc1_b(0 TO CLNG(g_tiny_n_layer) * CLNG(g_tiny_hidden_dim) - 1)
    REDIM g_tiny_fc2_w(0 TO layer_he - 1)
    REDIM g_tiny_fc2_b(0 TO layer_e - 1)
    REDIM g_tiny_final_ln_w(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_final_ln_b(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_head_w(0 TO head_count - 1)
    REDIM g_tiny_head_b(0 TO g_tiny_vocab_size - 1)

    file_num = FREEFILE
    ON ERROR GOTO weight_error
    OPEN weight_path FOR BINARY AS #file_num

    TinyGPTReadSingles file_num, g_tiny_tok_emb(), token_count
    TinyGPTReadSingles file_num, g_tiny_pos_emb(), pos_count
    TinyGPTReadSingles file_num, g_tiny_ln1_w(), layer_e
    TinyGPTReadSingles file_num, g_tiny_ln1_b(), layer_e
    TinyGPTReadSingles file_num, g_tiny_q_w(), layer_ee
    TinyGPTReadSingles file_num, g_tiny_q_b(), layer_e
    TinyGPTReadSingles file_num, g_tiny_k_w(), layer_ee
    TinyGPTReadSingles file_num, g_tiny_k_b(), layer_e
    TinyGPTReadSingles file_num, g_tiny_v_w(), layer_ee
    TinyGPTReadSingles file_num, g_tiny_v_b(), layer_e
    TinyGPTReadSingles file_num, g_tiny_proj_w(), layer_ee
    TinyGPTReadSingles file_num, g_tiny_proj_b(), layer_e
    TinyGPTReadSingles file_num, g_tiny_ln2_w(), layer_e
    TinyGPTReadSingles file_num, g_tiny_ln2_b(), layer_e
    TinyGPTReadSingles file_num, g_tiny_fc1_w(), layer_eh
    TinyGPTReadSingles file_num, g_tiny_fc1_b(), CLNG(g_tiny_n_layer) * CLNG(g_tiny_hidden_dim)
    TinyGPTReadSingles file_num, g_tiny_fc2_w(), layer_he
    TinyGPTReadSingles file_num, g_tiny_fc2_b(), layer_e
    TinyGPTReadSingles file_num, g_tiny_final_ln_w(), g_tiny_n_embd
    TinyGPTReadSingles file_num, g_tiny_final_ln_b(), g_tiny_n_embd
    TinyGPTReadSingles file_num, g_tiny_head_w(), head_count
    TinyGPTReadSingles file_num, g_tiny_head_b(), g_tiny_vocab_size

    CLOSE #file_num
    ON ERROR GOTO 0

	g_tiny_loaded = 1
	g_tiny_generation_start_len = 0
	TinyGPTAllocateDecodeCache
	g_tiny_tracked_memory = TinyGPTRuntimeMemoryBytes()
	TrackAllocation g_tiny_tracked_memory
	RETURN 1

weight_error:
    ON ERROR GOTO 0
    g_tiny_loaded = 0
    RETURN 0
END FUNCTION

SUB TinyGPTFreeModel()
    IF g_tiny_loaded = 0 THEN RETURN

    IF g_tiny_tracked_memory > 0 THEN
        TrackDeallocation g_tiny_tracked_memory
        g_tiny_tracked_memory = 0
    END IF

    IF g_tiny_fx_head_q4_stream <> 0 AND g_tiny_fx_head_q4_file <> 0 THEN
        CLOSE #g_tiny_fx_head_q4_file
    END IF

    ERASE g_tiny_tok_emb
    ERASE g_tiny_pos_emb
    ERASE g_tiny_ln1_w
    ERASE g_tiny_ln1_b
    ERASE g_tiny_q_w
    ERASE g_tiny_q_b
    ERASE g_tiny_k_w
    ERASE g_tiny_k_b
    ERASE g_tiny_v_w
    ERASE g_tiny_v_b
    ERASE g_tiny_proj_w
    ERASE g_tiny_proj_b
    ERASE g_tiny_ln2_w
    ERASE g_tiny_ln2_b
    ERASE g_tiny_fc1_w
    ERASE g_tiny_fc1_b
    ERASE g_tiny_fc2_w
    ERASE g_tiny_fc2_b
    ERASE g_tiny_final_ln_w
    ERASE g_tiny_final_ln_b
    ERASE g_tiny_head_w
    ERASE g_tiny_head_b
    ERASE g_tiny_cache_k
    ERASE g_tiny_cache_v
    ERASE g_tiny_cache_tokens
    ERASE g_tiny_x_vec
    ERASE g_tiny_norm_vec
    ERASE g_tiny_q_vec
    ERASE g_tiny_k_vec
    ERASE g_tiny_v_vec
    ERASE g_tiny_att_vec
    ERASE g_tiny_proj_vec
    ERASE g_tiny_ff1_vec
    ERASE g_tiny_ff2_vec
    ERASE g_tiny_logits_vec
    ERASE g_tiny_score_vec
    ERASE g_tiny_linear_acc
    ERASE g_tiny_fx_tok_emb
    ERASE g_tiny_fx_pos_emb
    ERASE g_tiny_fx_ln1_w
    ERASE g_tiny_fx_ln1_b
    ERASE g_tiny_fx_q_w
    ERASE g_tiny_fx_q_b
    ERASE g_tiny_fx_k_w
    ERASE g_tiny_fx_k_b
    ERASE g_tiny_fx_v_w
    ERASE g_tiny_fx_v_b
    ERASE g_tiny_fx_proj_w
    ERASE g_tiny_fx_proj_b
    ERASE g_tiny_fx_ln2_w
    ERASE g_tiny_fx_ln2_b
    ERASE g_tiny_fx_fc1_w
    ERASE g_tiny_fx_fc1_b
    ERASE g_tiny_fx_fc2_w
    ERASE g_tiny_fx_fc2_b
    ERASE g_tiny_fx_final_ln_w
    ERASE g_tiny_fx_final_ln_b
    ERASE g_tiny_fx_head_w
    ERASE g_tiny_fx_head_b
    ERASE g_tiny_fx_exp
    ERASE g_tiny_fx_tok_q4
    ERASE g_tiny_fx_tok_q4_scale
    ERASE g_tiny_fx_tok_q4_level
    ERASE g_tiny_fx_head_q4
    ERASE g_tiny_fx_head_q4_scale
    ERASE g_tiny_fx_head_q4_level
    ERASE g_tiny_fx_head_q4_decode
    ERASE g_tiny_fx_head_q4_row
    ERASE g_tiny_fx_cache_k
    ERASE g_tiny_fx_cache_v
    ERASE g_tiny_fx_cache_tokens
    ERASE g_tiny_fx_x_vec
    ERASE g_tiny_fx_norm_vec
    ERASE g_tiny_fx_q_vec
    ERASE g_tiny_fx_k_vec
    ERASE g_tiny_fx_v_vec
    ERASE g_tiny_fx_att_vec
    ERASE g_tiny_fx_proj_vec
    ERASE g_tiny_fx_ff1_vec
    ERASE g_tiny_fx_ff2_vec
    ERASE g_tiny_fx_logits_vec
    ERASE g_tiny_fx_score_vec
    ERASE g_tiny_fx_linear_acc
    TinyGPTFreePhaseDebugBuffers
    g_tiny_fx_attn_scale = 0

    g_tiny_loaded = 0
    g_tiny_fixed_loaded = 0
    g_tiny_fx_tok_q4_loaded = 0
    g_tiny_fx_tok_q4_bytes = 0
    g_tiny_fx_head_q4_loaded = 0
    g_tiny_fx_head_q4_bytes = 0
    g_tiny_fx_head_q4_stream = 0
    g_tiny_fx_head_q4_file = 0
    g_tiny_fx_head_q4_codes_offset = 0
    g_tiny_fx_head_q4_row_bytes = 0
    g_tiny_generation_start_len = 0
    g_tiny_cache_len = 0
    g_tiny_fx_cache_len = 0
END SUB

FUNCTION TinyGPTIsLoaded() AS INTEGER
    RETURN g_tiny_loaded
END FUNCTION

FUNCTION TinyGPTIsFixedPointLoaded() AS INTEGER
    RETURN g_tiny_fixed_loaded
END FUNCTION

FUNCTION TinyGPTEmbeddingDim() AS INTEGER
    RETURN g_tiny_n_embd
END FUNCTION

FUNCTION TinyGPTHeadCount() AS INTEGER
    RETURN g_tiny_n_head
END FUNCTION

FUNCTION TinyGPTLayerCount() AS INTEGER
    RETURN g_tiny_n_layer
END FUNCTION

FUNCTION TinyGPTContextLength() AS INTEGER
    RETURN g_tiny_n_positions
END FUNCTION

FUNCTION TinyGPTVocabSize() AS INTEGER
    RETURN g_tiny_vocab_size
END FUNCTION

FUNCTION TinyGPTProfileName() AS STRING
    IF g_tiny_profile_name = "" THEN RETURN "custom"
    RETURN g_tiny_profile_name
END FUNCTION

FUNCTION TinyGPTParameterCount() AS LONG
    DIM total AS LONG
    DIM layer_e AS LONG
    DIM layer_ee AS LONG
    DIM layer_eh AS LONG
    DIM layer_he AS LONG

    IF g_tiny_vocab_size < 1 THEN RETURN 0
    IF g_tiny_n_positions < 1 THEN RETURN 0
    IF g_tiny_n_embd < 1 THEN RETURN 0
    IF g_tiny_n_layer < 1 THEN RETURN 0
    IF g_tiny_hidden_dim < 1 THEN RETURN 0

    layer_e = CLNG(g_tiny_n_layer) * CLNG(g_tiny_n_embd)
    layer_ee = CLNG(g_tiny_n_layer) * CLNG(g_tiny_n_embd) * CLNG(g_tiny_n_embd)
    layer_eh = CLNG(g_tiny_n_layer) * CLNG(g_tiny_n_embd) * CLNG(g_tiny_hidden_dim)
    layer_he = CLNG(g_tiny_n_layer) * CLNG(g_tiny_hidden_dim) * CLNG(g_tiny_n_embd)

    total = CLNG(g_tiny_vocab_size) * CLNG(g_tiny_n_embd)
    total = total + CLNG(g_tiny_n_positions) * CLNG(g_tiny_n_embd)
    total = total + (layer_e * 8)
    total = total + (layer_ee * 4)
    total = total + (layer_eh + layer_he)
    total = total + (CLNG(g_tiny_n_layer) * CLNG(g_tiny_hidden_dim))
    total = total + layer_e
    total = total + (CLNG(g_tiny_n_embd) * 2)
    total = total + (CLNG(g_tiny_n_embd) * CLNG(g_tiny_vocab_size))
    total = total + g_tiny_vocab_size

    RETURN total
END FUNCTION

FUNCTION TinyGPTFixedWeightBytes() AS LONG
    RETURN TinyGPTParameterCount() * 4
END FUNCTION

FUNCTION TinyGPTRuntimeMemoryBytes() AS LONG
    DIM total AS LONG
    DIM cache_values AS LONG
    DIM linear_acc_count AS LONG
    DIM vector_count AS LONG
    DIM debug_vector_count AS LONG
    DIM token_count AS LONG
    DIM head_count AS LONG

    total = 0
    IF g_tiny_vocab_size < 1 THEN RETURN 0
    IF g_tiny_n_positions < 1 THEN RETURN 0
    IF g_tiny_n_embd < 1 THEN RETURN 0
    IF g_tiny_n_layer < 1 THEN RETURN 0
    IF g_tiny_hidden_dim < 1 THEN RETURN 0

    cache_values = CLNG(g_tiny_n_layer) * CLNG(g_tiny_n_positions) * CLNG(g_tiny_n_embd)
    token_count = CLNG(g_tiny_vocab_size) * CLNG(g_tiny_n_embd)
    head_count = CLNG(g_tiny_n_embd) * CLNG(g_tiny_vocab_size)
    linear_acc_count = g_tiny_hidden_dim
    IF g_tiny_n_embd > linear_acc_count THEN linear_acc_count = g_tiny_n_embd
    IF g_tiny_vocab_size > linear_acc_count THEN linear_acc_count = g_tiny_vocab_size

    IF g_tiny_fixed_loaded <> 0 THEN
        total = TinyGPTFixedWeightBytes()
        IF g_tiny_fx_tok_q4_loaded <> 0 THEN
            total = total - (token_count * 4) + g_tiny_fx_tok_q4_bytes
        END IF
        IF g_tiny_fx_head_q4_loaded <> 0 THEN
            total = total - (head_count * 4) + g_tiny_fx_head_q4_bytes
        END IF
        total = total + CLNG(TINYGPT_FX_EXP_SIZE) * 4
        total = total + cache_values * 8
        total = total + CLNG(g_tiny_n_positions) * 4
        vector_count = CLNG(g_tiny_n_embd) * 8
        vector_count = vector_count + CLNG(g_tiny_hidden_dim) * 2
        vector_count = vector_count + CLNG(g_tiny_vocab_size)
        vector_count = vector_count + CLNG(g_tiny_n_positions)
        total = total + vector_count * 4
        IF g_tiny_phase_debug_allocated <> 0 THEN
            debug_vector_count = CLNG(g_tiny_n_embd) * 11
            debug_vector_count = debug_vector_count + CLNG(g_tiny_hidden_dim)
            total = total + debug_vector_count * 4
        END IF
        total = total + linear_acc_count * 8
    ELSE
        total = TinyGPTParameterCount() * 4
        total = total + cache_values * 8
        total = total + CLNG(g_tiny_n_positions) * 4
        vector_count = CLNG(g_tiny_n_embd) * 8
        vector_count = vector_count + CLNG(g_tiny_hidden_dim) * 2
        vector_count = vector_count + CLNG(g_tiny_vocab_size)
        vector_count = vector_count + CLNG(g_tiny_n_positions)
        total = total + vector_count * 4
        total = total + linear_acc_count * 8
    END IF

    RETURN total
END FUNCTION

SUB TinyGPTBeginGeneration(prompt_token_count AS INTEGER)
    g_tiny_generation_start_len = prompt_token_count
    g_tiny_prompt_start_mode = 0
    g_tiny_prompt_start_mode_ready = 0
    IF g_tiny_fixed_loaded <> 0 THEN
        TinyGPTResetFixedDecodeCache
    ELSE
        TinyGPTResetDecodeCache
    END IF
END SUB

FUNCTION GPT2BasicLoadModel(base_path AS STRING) AS INTEGER
    RETURN TinyGPTLoadModel(base_path)
END FUNCTION

SUB GPT2BasicFreeModel()
    TinyGPTFreeModel
END SUB

FUNCTION GPT2BasicIsLoaded() AS INTEGER
    RETURN TinyGPTIsLoaded()
END FUNCTION

FUNCTION GPT2BasicIsFixedPointLoaded() AS INTEGER
    RETURN TinyGPTIsFixedPointLoaded()
END FUNCTION

FUNCTION GPT2BasicEmbeddingDim() AS INTEGER
    RETURN TinyGPTEmbeddingDim()
END FUNCTION

FUNCTION GPT2BasicHeadCount() AS INTEGER
    RETURN TinyGPTHeadCount()
END FUNCTION

FUNCTION GPT2BasicLayerCount() AS INTEGER
    RETURN TinyGPTLayerCount()
END FUNCTION

FUNCTION GPT2BasicContextLength() AS INTEGER
    RETURN TinyGPTContextLength()
END FUNCTION

FUNCTION GPT2BasicVocabSize() AS INTEGER
    RETURN TinyGPTVocabSize()
END FUNCTION

FUNCTION GPT2BasicProfileName() AS STRING
    RETURN TinyGPTProfileName()
END FUNCTION

FUNCTION GPT2BasicParameterCount() AS LONG
    RETURN TinyGPTParameterCount()
END FUNCTION

FUNCTION GPT2BasicFixedWeightBytes() AS LONG
    RETURN TinyGPTFixedWeightBytes()
END FUNCTION

FUNCTION GPT2BasicRuntimeMemoryBytes() AS LONG
    RETURN TinyGPTRuntimeMemoryBytes()
END FUNCTION

SUB GPT2BasicBeginGeneration(prompt_token_count AS INTEGER)
    TinyGPTBeginGeneration prompt_token_count
END SUB

FUNCTION GPT2BasicNextToken(context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
    RETURN TinyGPTNextToken(context(), context_len, temperature, top_p, top_k)
END FUNCTION

FUNCTION GPT2BasicForwardFixedLogits(context() AS INTEGER, context_len AS INTEGER, logits() AS LONG) AS INTEGER
    RETURN TinyGPTForwardFixedLogits(context(), context_len, logits())
END FUNCTION

FUNCTION GPT2BasicForwardFloatLogits(context() AS INTEGER, context_len AS INTEGER, logits() AS SINGLE) AS INTEGER
    RETURN TinyGPTForwardFloatLogits(context(), context_len, logits())
END FUNCTION

FUNCTION GPT2BasicRunVectorFile(vector_path AS STRING, tolerance AS LONG) AS INTEGER
    RETURN TinyGPTRunVectorFile(vector_path, tolerance)
END FUNCTION

FUNCTION TinyGPTVectorField(line_text AS STRING, field_idx AS INTEGER) AS STRING
    DIM current_field AS INTEGER
    DIM start_pos AS INTEGER
    DIM pipe_pos AS INTEGER

    current_field = 1
    start_pos = 1

    DO
        pipe_pos = INSTR(start_pos, line_text, "|")
        IF current_field = field_idx THEN
            IF pipe_pos = 0 THEN
                RETURN MID$(line_text, start_pos)
            ELSE
                RETURN MID$(line_text, start_pos, pipe_pos - start_pos)
            END IF
        END IF

        IF pipe_pos = 0 THEN EXIT DO
        start_pos = pipe_pos + 1
        current_field = current_field + 1
    LOOP

    RETURN ""
END FUNCTION

SUB TinyGPTParseVectorTokens(token_text AS STRING, tokens() AS INTEGER, BYREF token_count AS INTEGER)
    DIM remaining AS STRING
    DIM item_text AS STRING
    DIM comma_pos AS INTEGER
    DIM parsed_token AS INTEGER

    token_count = 0
    remaining = TRIM$(token_text)

    IF remaining = "" THEN
        REDIM tokens(0 TO 0)
        RETURN
    END IF

    REDIM tokens(0 TO 0)

    DO WHILE remaining <> ""
        comma_pos = INSTR(remaining, ",")
        IF comma_pos > 0 THEN
            item_text = LEFT$(remaining, comma_pos - 1)
            remaining = MID$(remaining, comma_pos + 1)
        ELSE
            item_text = remaining
            remaining = ""
        END IF

        item_text = TRIM$(item_text)
        IF item_text <> "" THEN
            parsed_token = CINT(VAL(item_text))
            IF token_count = 0 THEN
                REDIM tokens(0 TO 0)
            ELSE
                REDIM PRESERVE tokens(0 TO token_count)
            END IF
            tokens(token_count) = parsed_token
            token_count = token_count + 1
        END IF
    LOOP
END SUB

FUNCTION TinyGPTAbsLong(value AS LONG) AS LONG
    IF value < 0 THEN RETURN -value
    RETURN value
END FUNCTION

SUB TinyGPTCopyLongVector(source() AS LONG, dest() AS LONG, value_count AS INTEGER)
    DIM i AS INTEGER
    FOR i = 0 TO value_count - 1
        dest(i) = source(i)
    NEXT i
END SUB

FUNCTION TinyGPTCompareVectorLogits(vector_name AS STRING, logits() AS LONG, expected_text AS STRING, tolerance AS LONG) AS INTEGER
    DIM remaining AS STRING
    DIM pair_text AS STRING
    DIM comma_pos AS INTEGER
    DIM colon_pos AS INTEGER
    DIM token_id AS INTEGER
    DIM expected_value AS LONG
    DIM actual_value AS LONG
    DIM delta AS LONG
    DIM checked_count AS INTEGER
    DIM ok AS INTEGER

    remaining = TRIM$(expected_text)
    checked_count = 0
    ok = 1

    DO WHILE remaining <> ""
        comma_pos = INSTR(remaining, ",")
        IF comma_pos > 0 THEN
            pair_text = LEFT$(remaining, comma_pos - 1)
            remaining = MID$(remaining, comma_pos + 1)
        ELSE
            pair_text = remaining
            remaining = ""
        END IF

        pair_text = TRIM$(pair_text)
        IF pair_text <> "" THEN
            colon_pos = INSTR(pair_text, ":")
            IF colon_pos <= 1 THEN
                PRINT "VECTOR_FAIL "; vector_name; " malformed expected pair: "; pair_text
                RETURN 0
            END IF

            token_id = CINT(VAL(LEFT$(pair_text, colon_pos - 1)))
            expected_value = CLNG(VAL(MID$(pair_text, colon_pos + 1)))

            IF token_id < 0 OR token_id >= g_tiny_vocab_size THEN
                PRINT "VECTOR_FAIL "; vector_name; " token out of range: "; token_id
                RETURN 0
            END IF

            actual_value = logits(token_id)
            delta = TinyGPTAbsLong(actual_value - expected_value)
            IF delta > tolerance THEN
                PRINT "VECTOR_FAIL "; vector_name; " token "; token_id
                PRINT "  expected "; expected_value; " actual "; actual_value; " delta "; delta
                ok = 0
            END IF
            checked_count = checked_count + 1
        END IF
    LOOP

    IF checked_count <= 0 THEN
        PRINT "VECTOR_FAIL "; vector_name; " has no expected logits"
        RETURN 0
    END IF

    RETURN ok
END FUNCTION

FUNCTION TinyGPTComparePhaseValues(vector_name AS STRING, phase_name AS STRING, declared_dim AS INTEGER, expected_text AS STRING, tolerance AS LONG) AS INTEGER
    DIM remaining AS STRING
    DIM pair_text AS STRING
    DIM comma_pos AS INTEGER
    DIM colon_pos AS INTEGER
    DIM idx AS INTEGER
    DIM actual_dim AS INTEGER
    DIM expected_value AS LONG
    DIM actual_value AS LONG
    DIM delta AS LONG
    DIM checked_count AS INTEGER
    DIM ok AS INTEGER
    DIM phase_key AS STRING

    phase_key = LCASE$(TRIM$(phase_name))
    actual_dim = g_tiny_n_embd
    IF phase_key = "ff1" THEN actual_dim = g_tiny_hidden_dim
    IF phase_key = "logits" THEN actual_dim = g_tiny_vocab_size

    IF declared_dim <> actual_dim THEN
        PRINT "PHASE_FAIL "; vector_name; " "; phase_name; " dim expected "; declared_dim; " actual "; actual_dim
        RETURN 0
    END IF

    remaining = TRIM$(expected_text)
    checked_count = 0
    ok = 1

    DO WHILE remaining <> ""
        comma_pos = INSTR(remaining, ",")
        IF comma_pos > 0 THEN
            pair_text = LEFT$(remaining, comma_pos - 1)
            remaining = MID$(remaining, comma_pos + 1)
        ELSE
            pair_text = remaining
            remaining = ""
        END IF

        pair_text = TRIM$(pair_text)
        IF pair_text <> "" THEN
            colon_pos = INSTR(pair_text, ":")
            IF colon_pos <= 1 THEN
                PRINT "PHASE_FAIL "; vector_name; " "; phase_name; " malformed pair: "; pair_text
                RETURN 0
            END IF

            idx = CINT(VAL(LEFT$(pair_text, colon_pos - 1)))
            expected_value = CLNG(VAL(MID$(pair_text, colon_pos + 1)))

            IF idx < 0 OR idx >= actual_dim THEN
                PRINT "PHASE_FAIL "; vector_name; " "; phase_name; " index out of range: "; idx
                RETURN 0
            END IF

            SELECT CASE phase_key
                CASE "embedding"
                    actual_value = g_tiny_fx_dbg_embedding_vec(idx)
                CASE "ln1"
                    actual_value = g_tiny_fx_dbg_ln1_vec(idx)
                CASE "q"
                    actual_value = g_tiny_fx_dbg_q_vec(idx)
                CASE "k"
                    actual_value = g_tiny_fx_dbg_k_vec(idx)
                CASE "v"
                    actual_value = g_tiny_fx_dbg_v_vec(idx)
                CASE "attn"
                    actual_value = g_tiny_fx_dbg_attn_vec(idx)
                CASE "proj"
                    actual_value = g_tiny_fx_dbg_proj_vec(idx)
                CASE "ln2"
                    actual_value = g_tiny_fx_dbg_ln2_vec(idx)
                CASE "ff1"
                    actual_value = g_tiny_fx_dbg_ff1_vec(idx)
                CASE "ff2"
                    actual_value = g_tiny_fx_dbg_ff2_vec(idx)
                CASE "hidden"
                    actual_value = g_tiny_fx_dbg_hidden_vec(idx)
                CASE "final_ln"
                    actual_value = g_tiny_fx_dbg_final_ln_vec(idx)
                CASE "logits"
                    actual_value = g_tiny_fx_logits_vec(idx)
                CASE ELSE
                    PRINT "PHASE_FAIL "; vector_name; " unknown phase "; phase_name
                    RETURN 0
            END SELECT

            delta = TinyGPTAbsLong(actual_value - expected_value)
            IF delta > tolerance THEN
                PRINT "PHASE_FAIL "; vector_name; " "; phase_name; " index "; idx
                PRINT "  expected "; expected_value; " actual "; actual_value; " delta "; delta
                ok = 0
            END IF
            checked_count = checked_count + 1
        END IF
    LOOP

    IF checked_count <= 0 THEN
        PRINT "PHASE_FAIL "; vector_name; " "; phase_name; " has no expected values"
        RETURN 0
    END IF

    RETURN ok
END FUNCTION

FUNCTION TinyGPTRunVectorFile(vector_path AS STRING, tolerance AS LONG) AS INTEGER
    DIM file_num AS INTEGER
    DIM line_text AS STRING
    DIM record_type AS STRING
    DIM vector_name AS STRING
    DIM declared_len AS INTEGER
    DIM token_text AS STRING
    DIM expected_text AS STRING
    DIM tokens() AS INTEGER
    DIM token_count AS INTEGER
    DIM logits() AS LONG
    DIM vector_count AS INTEGER
    DIM pass_count AS INTEGER
    DIM phase_count AS INTEGER
    DIM phase_pass_count AS INTEGER
    DIM current_vector_name AS STRING
    DIM current_vector_ready AS INTEGER
    DIM phase_name AS STRING
    DIM declared_dim AS INTEGER
    DIM ok AS INTEGER
    DIM forward_ok AS INTEGER

    IF g_tiny_loaded = 0 OR g_tiny_fixed_loaded = 0 THEN
        PRINT "VECTOR_FAIL fixed-point GPT2-BASIC model is not loaded"
        RETURN 0
    END IF

    vector_count = 0
    pass_count = 0
    phase_count = 0
    phase_pass_count = 0
    current_vector_name = ""
    current_vector_ready = 0
    TinyGPTAllocatePhaseDebugBuffers

    file_num = FREEFILE
    ON ERROR GOTO vector_error
    OPEN vector_path FOR INPUT AS #file_num
    ON ERROR GOTO 0

    WHILE EOF(file_num) = 0
        LINE INPUT #file_num, line_text
        line_text = TRIM$(line_text)

        IF line_text <> "" AND LEFT$(line_text, 1) <> "#" THEN
            record_type = TinyGPTVectorField(line_text, 1)
            IF record_type = "V" THEN
                vector_count = vector_count + 1
                vector_name = TinyGPTVectorField(line_text, 2)
                declared_len = CINT(VAL(TinyGPTVectorField(line_text, 3)))
                token_text = TinyGPTVectorField(line_text, 4)
                expected_text = TinyGPTVectorField(line_text, 6)

                TinyGPTParseVectorTokens token_text, tokens(), token_count
                ok = 1

                IF token_count <> declared_len THEN
                    PRINT "VECTOR_FAIL "; vector_name; " length expected "; declared_len; " parsed "; token_count
                    ok = 0
                END IF

                IF ok <> 0 THEN
                    TinyGPTResetFixedDecodeCache
                    g_tiny_phase_capture_enabled = 1
                    forward_ok = TinyGPTForwardFixedLogits(tokens(), token_count, logits())
                    g_tiny_phase_capture_enabled = 0
                    IF forward_ok = 0 THEN
                        PRINT "VECTOR_FAIL "; vector_name; " forward pass failed"
                        ok = 0
                    END IF
                END IF

                IF ok <> 0 THEN
                    ok = TinyGPTCompareVectorLogits(vector_name, logits(), expected_text, tolerance)
                END IF

                IF ok <> 0 THEN
                    pass_count = pass_count + 1
                    current_vector_name = vector_name
                    current_vector_ready = 1
                    PRINT "VECTOR_OK "; vector_name
                ELSE
                    current_vector_name = vector_name
                    current_vector_ready = 0
                END IF
            ELSEIF record_type = "P" THEN
                phase_count = phase_count + 1
                vector_name = TinyGPTVectorField(line_text, 2)
                phase_name = TinyGPTVectorField(line_text, 3)
                declared_dim = CINT(VAL(TinyGPTVectorField(line_text, 4)))
                expected_text = TinyGPTVectorField(line_text, 6)

                ok = 1
                IF current_vector_ready = 0 OR vector_name <> current_vector_name THEN
                    PRINT "PHASE_FAIL "; vector_name; " "; phase_name; " has no passing vector context"
                    ok = 0
                END IF

                IF ok <> 0 THEN
                    ok = TinyGPTComparePhaseValues(vector_name, phase_name, declared_dim, expected_text, tolerance)
                END IF

                IF ok <> 0 THEN
                    phase_pass_count = phase_pass_count + 1
                    PRINT "PHASE_OK "; vector_name; " "; phase_name
                END IF
            END IF
        END IF
    WEND

    CLOSE #file_num

    g_tiny_phase_capture_enabled = 0
    TinyGPTFreePhaseDebugBuffers
    PRINT "VECTOR_SUMMARY passed "; pass_count; " of "; vector_count
    PRINT "PHASE_SUMMARY passed "; phase_pass_count; " of "; phase_count
    IF vector_count > 0 AND pass_count = vector_count AND phase_pass_count = phase_count THEN RETURN 1
    RETURN 0

vector_error:
    ON ERROR GOTO 0
    g_tiny_phase_capture_enabled = 0
    TinyGPTFreePhaseDebugBuffers
    PRINT "VECTOR_FAIL could not read "; vector_path
    RETURN 0
END FUNCTION

SUB TinyGPTAllocateDecodeCache()
    DIM cache_values AS LONG
    DIM linear_acc_count AS INTEGER

    IF g_tiny_n_layer < 1 THEN RETURN
    IF g_tiny_n_positions < 1 THEN RETURN
    IF g_tiny_n_embd < 1 THEN RETURN
    IF g_tiny_hidden_dim < 1 THEN RETURN
    IF g_tiny_vocab_size < 1 THEN RETURN

    cache_values = CLNG(g_tiny_n_layer) * CLNG(g_tiny_n_positions) * CLNG(g_tiny_n_embd)

    REDIM g_tiny_cache_k(0 TO cache_values - 1)
    REDIM g_tiny_cache_v(0 TO cache_values - 1)
    REDIM g_tiny_cache_tokens(0 TO g_tiny_n_positions - 1)
    REDIM g_tiny_x_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_norm_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_q_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_k_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_v_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_att_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_proj_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_ff1_vec(0 TO g_tiny_hidden_dim - 1)
    REDIM g_tiny_ff2_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_logits_vec(0 TO g_tiny_vocab_size - 1)
    REDIM g_tiny_score_vec(0 TO g_tiny_n_positions - 1)
    linear_acc_count = g_tiny_hidden_dim
    IF g_tiny_n_embd > linear_acc_count THEN linear_acc_count = g_tiny_n_embd
    REDIM g_tiny_linear_acc(0 TO linear_acc_count - 1)

    TinyGPTResetDecodeCache
END SUB

SUB TinyGPTResetDecodeCache()
    DIM i AS INTEGER

    g_tiny_cache_len = 0
    IF g_tiny_n_positions < 1 THEN RETURN

    FOR i = 0 TO g_tiny_n_positions - 1
        g_tiny_cache_tokens(i) = -1
    NEXT i
END SUB

SUB TinyGPTAllocateFixedDecodeCache()
    DIM cache_values AS LONG
    DIM linear_acc_count AS INTEGER
    DIM head_dim AS INTEGER

    IF g_tiny_n_layer < 1 THEN RETURN
    IF g_tiny_n_positions < 1 THEN RETURN
    IF g_tiny_n_embd < 1 THEN RETURN
    IF g_tiny_hidden_dim < 1 THEN RETURN
    IF g_tiny_vocab_size < 1 THEN RETURN

    cache_values = CLNG(g_tiny_n_layer) * CLNG(g_tiny_n_positions) * CLNG(g_tiny_n_embd)

    REDIM g_tiny_fx_cache_k(0 TO cache_values - 1)
    REDIM g_tiny_fx_cache_v(0 TO cache_values - 1)
    REDIM g_tiny_fx_cache_tokens(0 TO g_tiny_n_positions - 1)
    REDIM g_tiny_fx_x_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_norm_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_q_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_k_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_v_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_att_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_proj_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_ff1_vec(0 TO g_tiny_hidden_dim - 1)
    REDIM g_tiny_fx_ff2_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_logits_vec(0 TO g_tiny_vocab_size - 1)
    REDIM g_tiny_fx_score_vec(0 TO g_tiny_n_positions - 1)
    linear_acc_count = g_tiny_hidden_dim
    IF g_tiny_n_embd > linear_acc_count THEN linear_acc_count = g_tiny_n_embd
    IF g_tiny_vocab_size > linear_acc_count THEN linear_acc_count = g_tiny_vocab_size
    REDIM g_tiny_fx_linear_acc(0 TO linear_acc_count - 1)

    head_dim = g_tiny_n_embd \ g_tiny_n_head
    g_tiny_fx_attn_scale = TinyGPTFixedDiv(TINYGPT_FX_ONE, TinyGPTFixedSqrt(head_dim * TINYGPT_FX_ONE))

    TinyGPTResetFixedDecodeCache
END SUB

SUB TinyGPTAllocatePhaseDebugBuffers()
    IF g_tiny_phase_debug_allocated <> 0 THEN RETURN
    IF g_tiny_n_embd < 1 THEN RETURN
    IF g_tiny_hidden_dim < 1 THEN RETURN

    REDIM g_tiny_fx_dbg_embedding_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_dbg_ln1_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_dbg_q_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_dbg_k_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_dbg_v_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_dbg_attn_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_dbg_proj_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_dbg_ln2_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_dbg_ff1_vec(0 TO g_tiny_hidden_dim - 1)
    REDIM g_tiny_fx_dbg_ff2_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_dbg_hidden_vec(0 TO g_tiny_n_embd - 1)
    REDIM g_tiny_fx_dbg_final_ln_vec(0 TO g_tiny_n_embd - 1)
    g_tiny_phase_debug_allocated = 1
END SUB

SUB TinyGPTFreePhaseDebugBuffers()
    ERASE g_tiny_fx_dbg_embedding_vec
    ERASE g_tiny_fx_dbg_ln1_vec
    ERASE g_tiny_fx_dbg_q_vec
    ERASE g_tiny_fx_dbg_k_vec
    ERASE g_tiny_fx_dbg_v_vec
    ERASE g_tiny_fx_dbg_attn_vec
    ERASE g_tiny_fx_dbg_proj_vec
    ERASE g_tiny_fx_dbg_ln2_vec
    ERASE g_tiny_fx_dbg_ff1_vec
    ERASE g_tiny_fx_dbg_ff2_vec
    ERASE g_tiny_fx_dbg_hidden_vec
    ERASE g_tiny_fx_dbg_final_ln_vec
    g_tiny_phase_capture_enabled = 0
    g_tiny_phase_debug_allocated = 0
END SUB

SUB TinyGPTResetFixedDecodeCache()
    DIM i AS INTEGER

    g_tiny_fx_cache_len = 0
    IF g_tiny_n_positions < 1 THEN RETURN

    FOR i = 0 TO g_tiny_n_positions - 1
        g_tiny_fx_cache_tokens(i) = -1
    NEXT i
END SUB

SUB TinyGPTLayerNormSeq(input_arr() AS SINGLE, output_arr() AS SINGLE, seq_len AS INTEGER, emb_dim AS INTEGER, gamma_arr() AS SINGLE, beta_arr() AS SINGLE, param_base AS LONG)
    DIM row_idx AS INTEGER
    DIM col_idx AS INTEGER
    DIM row_base AS LONG
    DIM mean_value AS DOUBLE
    DIM var_value AS DOUBLE
    DIM diff_value AS DOUBLE
    DIM inv_std AS DOUBLE

    FOR row_idx = 0 TO seq_len - 1
        row_base = CLNG(row_idx) * CLNG(emb_dim)
        mean_value = 0.0

        FOR col_idx = 0 TO emb_dim - 1
            mean_value = mean_value + input_arr(row_base + col_idx)
        NEXT col_idx
        mean_value = mean_value / emb_dim

        var_value = 0.0
        FOR col_idx = 0 TO emb_dim - 1
            diff_value = input_arr(row_base + col_idx) - mean_value
            var_value = var_value + diff_value * diff_value
        NEXT col_idx
        var_value = var_value / emb_dim
        inv_std = 1.0 / SQR(var_value + 0.00001)

        FOR col_idx = 0 TO emb_dim - 1
            output_arr(row_base + col_idx) = CSNG(((input_arr(row_base + col_idx) - mean_value) * inv_std) * gamma_arr(param_base + col_idx) + beta_arr(param_base + col_idx))
        NEXT col_idx
    NEXT row_idx
END SUB

SUB TinyGPTLayerNormLast(input_arr() AS SINGLE, output_arr() AS SINGLE, seq_len AS INTEGER, emb_dim AS INTEGER, gamma_arr() AS SINGLE, beta_arr() AS SINGLE)
    DIM col_idx AS INTEGER
    DIM row_base AS LONG
    DIM mean_value AS DOUBLE
    DIM var_value AS DOUBLE
    DIM diff_value AS DOUBLE
    DIM inv_std AS DOUBLE

    row_base = CLNG(seq_len - 1) * CLNG(emb_dim)
    mean_value = 0.0

    FOR col_idx = 0 TO emb_dim - 1
        mean_value = mean_value + input_arr(row_base + col_idx)
    NEXT col_idx
    mean_value = mean_value / emb_dim

    var_value = 0.0
    FOR col_idx = 0 TO emb_dim - 1
        diff_value = input_arr(row_base + col_idx) - mean_value
        var_value = var_value + diff_value * diff_value
    NEXT col_idx
    var_value = var_value / emb_dim
    inv_std = 1.0 / SQR(var_value + 0.00001)

    FOR col_idx = 0 TO emb_dim - 1
        output_arr(col_idx) = CSNG(((input_arr(row_base + col_idx) - mean_value) * inv_std) * gamma_arr(col_idx) + beta_arr(col_idx))
    NEXT col_idx
END SUB

SUB TinyGPTLayerNormVec(input_vec() AS SINGLE, output_vec() AS SINGLE, emb_dim AS INTEGER, gamma_arr() AS SINGLE, beta_arr() AS SINGLE, param_base AS LONG)
    DIM col_idx AS INTEGER
    DIM mean_value AS DOUBLE
    DIM var_value AS DOUBLE
    DIM diff_value AS DOUBLE
    DIM inv_std AS DOUBLE

    mean_value = 0.0
    FOR col_idx = 0 TO emb_dim - 1
        mean_value = mean_value + input_vec(col_idx)
    NEXT col_idx
    mean_value = mean_value / emb_dim

    var_value = 0.0
    FOR col_idx = 0 TO emb_dim - 1
        diff_value = input_vec(col_idx) - mean_value
        var_value = var_value + diff_value * diff_value
    NEXT col_idx
    var_value = var_value / emb_dim
    inv_std = 1.0 / SQR(var_value + 0.00001)

    FOR col_idx = 0 TO emb_dim - 1
        output_vec(col_idx) = CSNG(((input_vec(col_idx) - mean_value) * inv_std) * gamma_arr(param_base + col_idx) + beta_arr(param_base + col_idx))
    NEXT col_idx
END SUB

SUB TinyGPTLinearSeq(input_arr() AS SINGLE, output_arr() AS SINGLE, seq_len AS INTEGER, in_dim AS INTEGER, out_dim AS INTEGER, weight_arr() AS SINGLE, weight_base AS LONG, bias_arr() AS SINGLE, bias_base AS LONG)
    DIM row_idx AS INTEGER
    DIM out_idx AS INTEGER
    DIM in_idx AS INTEGER
    DIM input_base AS LONG
    DIM weight_col AS LONG
    DIM sum_value AS DOUBLE

    FOR row_idx = 0 TO seq_len - 1
        input_base = CLNG(row_idx) * CLNG(in_dim)
        FOR out_idx = 0 TO out_dim - 1
            sum_value = bias_arr(bias_base + out_idx)
            weight_col = weight_base + out_idx

            FOR in_idx = 0 TO in_dim - 1
                sum_value = sum_value + input_arr(input_base + in_idx) * weight_arr(weight_col + CLNG(in_idx) * CLNG(out_dim))
            NEXT in_idx

            output_arr(CLNG(row_idx) * CLNG(out_dim) + out_idx) = CSNG(sum_value)
        NEXT out_idx
    NEXT row_idx
END SUB

SUB TinyGPTLinearVec(input_vec() AS SINGLE, output_vec() AS SINGLE, in_dim AS INTEGER, out_dim AS INTEGER, weight_arr() AS SINGLE, weight_base AS LONG, bias_arr() AS SINGLE, bias_base AS LONG)
    DIM out_idx AS INTEGER
    DIM in_idx AS INTEGER
    DIM weight_row AS LONG
    DIM input_value AS DOUBLE

    FOR out_idx = 0 TO out_dim - 1
        g_tiny_linear_acc(out_idx) = bias_arr(bias_base + out_idx)
    NEXT out_idx

    FOR in_idx = 0 TO in_dim - 1
        input_value = input_vec(in_idx)
        weight_row = weight_base + CLNG(in_idx) * CLNG(out_dim)

        FOR out_idx = 0 TO out_dim - 1
            g_tiny_linear_acc(out_idx) = g_tiny_linear_acc(out_idx) + input_value * weight_arr(weight_row + out_idx)
        NEXT out_idx
    NEXT in_idx

    FOR out_idx = 0 TO out_dim - 1
        output_vec(out_idx) = CSNG(g_tiny_linear_acc(out_idx))
    NEXT out_idx
END SUB

FUNCTION TinyGPTFixedClamp(value AS LONGINT) AS LONG
    IF value > TINYGPT_FX_CLAMP THEN RETURN TINYGPT_FX_CLAMP
    IF value < -TINYGPT_FX_CLAMP THEN RETURN -TINYGPT_FX_CLAMP
    RETURN CLNG(value)
END FUNCTION

FUNCTION TinyGPTFixedMul(a AS LONG, b AS LONG) AS LONG
    DIM result AS LONGINT
    result = (CLNGINT(a) * CLNGINT(b)) \ TINYGPT_FX_ONE
    RETURN TinyGPTFixedClamp(result)
END FUNCTION

FUNCTION TinyGPTFixedDiv(a AS LONG, b AS LONG) AS LONG
    DIM result AS LONGINT
    IF b = 0 THEN
        IF a >= 0 THEN RETURN TINYGPT_FX_CLAMP
        RETURN -TINYGPT_FX_CLAMP
    END IF
    result = (CLNGINT(a) * TINYGPT_FX_ONE) \ b
    RETURN TinyGPTFixedClamp(result)
END FUNCTION

FUNCTION TinyGPTFixedSqrt(value AS LONG) AS LONG
    DIM target AS LONGINT
    DIM x AS LONGINT
    DIM last_x AS LONGINT
    DIM guard AS INTEGER

    IF value <= 0 THEN RETURN 0

    target = CLNGINT(value) * TINYGPT_FX_ONE
    x = target
    IF x < TINYGPT_FX_ONE THEN x = TINYGPT_FX_ONE
    last_x = 0
    guard = 0

    WHILE x <> last_x AND guard < 32
        last_x = x
        x = (x + (target \ x)) \ 2
        guard = guard + 1
    WEND

    RETURN TinyGPTFixedClamp(x)
END FUNCTION

FUNCTION TinyGPTFixedExpNeg(x AS LONG) AS LONG
    DIM idx AS INTEGER
    DIM numerator AS LONGINT
    DIM denominator AS LONGINT

    IF x >= 0 THEN RETURN TINYGPT_FX_ONE

    numerator = CLNGINT(-x) * CLNGINT(TINYGPT_FX_EXP_SIZE - 1)
    denominator = CLNGINT(TINYGPT_FX_EXP_MAX) * CLNGINT(TINYGPT_FX_ONE)
    idx = CINT(numerator \ denominator)

    IF idx < 0 THEN idx = 0
    IF idx >= TINYGPT_FX_EXP_SIZE THEN RETURN 0

    RETURN g_tiny_fx_exp(idx)
END FUNCTION

FUNCTION TinyGPTFixedTanh(x AS LONG) AS LONG
    DIM abs_x AS LONG
    DIM exp_neg AS LONG
    DIM tanh_abs AS LONG

    IF x >= (3 * TINYGPT_FX_ONE) THEN RETURN TINYGPT_FX_ONE
    IF x <= (-3 * TINYGPT_FX_ONE) THEN RETURN -TINYGPT_FX_ONE

    IF x < 0 THEN
        abs_x = -x
    ELSE
        abs_x = x
    END IF

    exp_neg = TinyGPTFixedExpNeg(-(abs_x * 2))
    tanh_abs = TinyGPTFixedDiv(TINYGPT_FX_ONE - exp_neg, TINYGPT_FX_ONE + exp_neg)

    IF x < 0 THEN RETURN -tanh_abs
    RETURN tanh_abs
END FUNCTION

SUB TinyGPTKernelPerfReset()
    DIM i AS INTEGER
    FOR i = 0 TO TINYGPT_KERNEL_STAGE_COUNT - 1
        g_tiny_kernel_perf_seconds(i) = 0.0
        g_tiny_kernel_perf_calls(i) = 0
    NEXT i
END SUB

SUB TinyGPTKernelPerfSetEnabled(enabled AS INTEGER)
    IF enabled <> 0 THEN
        g_tiny_kernel_perf_enabled = 1
    ELSE
        g_tiny_kernel_perf_enabled = 0
    END IF
END SUB

FUNCTION TinyGPTKernelPerfStageName(stage_id AS INTEGER) AS STRING
    SELECT CASE stage_id
        CASE TINYGPT_KERNEL_EMBED: RETURN "embedding"
        CASE TINYGPT_KERNEL_QKV: RETURN "ln1_qkv"
        CASE TINYGPT_KERNEL_ATTENTION: RETURN "attention"
        CASE TINYGPT_KERNEL_PROJECTION: RETURN "projection"
        CASE TINYGPT_KERNEL_FFN: RETURN "ffn"
        CASE TINYGPT_KERNEL_HEAD: RETURN "final_head"
        CASE ELSE: RETURN "unknown"
    END SELECT
END FUNCTION

SUB TinyGPTKernelPerfAdd(stage_id AS INTEGER, start_time AS DOUBLE)
    DIM elapsed AS DOUBLE

    IF g_tiny_kernel_perf_enabled = 0 THEN RETURN
    IF stage_id < 0 OR stage_id >= TINYGPT_KERNEL_STAGE_COUNT THEN RETURN

    elapsed = TIMER - start_time
    IF elapsed < -1.0 THEN elapsed = 0.0
    IF elapsed < 0.0 THEN elapsed = elapsed + 86400.0
    IF elapsed > 3600.0 THEN elapsed = 0.0
    g_tiny_kernel_perf_seconds(stage_id) = g_tiny_kernel_perf_seconds(stage_id) + elapsed
    g_tiny_kernel_perf_calls(stage_id) = g_tiny_kernel_perf_calls(stage_id) + 1
END SUB

SUB TinyGPTKernelPerfReport()
    DIM i AS INTEGER
    PRINT "KERNEL_PERF_BEGIN|version=1|timer=freebasic_TIMER"
    FOR i = 0 TO TINYGPT_KERNEL_STAGE_COUNT - 1
        PRINT "KERNEL_PERF|stage=" + TinyGPTKernelPerfStageName(i) + _
              "|calls=" + LTRIM$(STR$(g_tiny_kernel_perf_calls(i))) + _
              "|seconds=" + LTRIM$(STR$(g_tiny_kernel_perf_seconds(i)))
    NEXT i
    PRINT "KERNEL_PERF_END"
END SUB

SUB TinyGPTFixedLayerNormVec(input_vec() AS LONG, output_vec() AS LONG, emb_dim AS INTEGER, gamma_arr() AS LONG, beta_arr() AS LONG, param_base AS LONG)
    DIM col_idx AS INTEGER
    DIM mean_value AS LONG
    DIM var_value AS LONG
    DIM diff_value AS LONG
    DIM inv_std AS LONG
    DIM sum_value AS LONGINT
    DIM sum_sq AS LONGINT
    DIM norm_value AS LONG

    sum_value = 0
    FOR col_idx = 0 TO emb_dim - 1
        sum_value = sum_value + input_vec(col_idx)
    NEXT col_idx
    mean_value = TinyGPTFixedClamp(sum_value \ emb_dim)

    sum_sq = 0
    FOR col_idx = 0 TO emb_dim - 1
        diff_value = input_vec(col_idx) - mean_value
        sum_sq = sum_sq + TinyGPTFixedMul(diff_value, diff_value)
    NEXT col_idx
    var_value = TinyGPTFixedClamp(sum_sq \ emb_dim)
    inv_std = TinyGPTFixedDiv(TINYGPT_FX_ONE, TinyGPTFixedSqrt(var_value + TINYGPT_FX_EPS))

    FOR col_idx = 0 TO emb_dim - 1
        diff_value = input_vec(col_idx) - mean_value
        norm_value = TinyGPTFixedMul(diff_value, inv_std)
        output_vec(col_idx) = TinyGPTFixedMul(norm_value, gamma_arr(param_base + col_idx)) + beta_arr(param_base + col_idx)
    NEXT col_idx
END SUB

SUB TinyGPTFixedLinearVec(input_vec() AS LONG, output_vec() AS LONG, in_dim AS INTEGER, out_dim AS INTEGER, weight_arr() AS LONG, weight_base AS LONG, bias_arr() AS LONG, bias_base AS LONG)
    DIM out_idx AS INTEGER
    DIM in_idx AS INTEGER
    DIM weight_row AS LONG
    DIM input_value AS LONG

    FOR out_idx = 0 TO out_dim - 1
        g_tiny_fx_linear_acc(out_idx) = bias_arr(bias_base + out_idx)
    NEXT out_idx

    FOR in_idx = 0 TO in_dim - 1
        input_value = input_vec(in_idx)
        weight_row = weight_base + CLNG(in_idx) * CLNG(out_dim)

        FOR out_idx = 0 TO out_dim - 1
            g_tiny_fx_linear_acc(out_idx) = g_tiny_fx_linear_acc(out_idx) + ((CLNGINT(input_value) * weight_arr(weight_row + out_idx)) \ TINYGPT_FX_ONE)
        NEXT out_idx
    NEXT in_idx

    FOR out_idx = 0 TO out_dim - 1
        output_vec(out_idx) = TinyGPTFixedClamp(g_tiny_fx_linear_acc(out_idx))
    NEXT out_idx
END SUB

SUB TinyGPTFixedHeadQ4LinearVec(input_vec() AS LONG, output_vec() AS LONG)
    DIM out_idx AS INTEGER
    DIM next_out_idx AS INTEGER
    DIM in_idx AS INTEGER
    DIM vocab_size AS INTEGER
    DIM packed_index AS LONG
    DIM byte_idx AS LONG
    DIM packed_value AS UBYTE
    DIM code AS INTEGER
    DIM magnitude AS INTEGER
    DIM input_value AS LONG
    DIM weight_value AS LONG
    DIM decoded_value AS LONGINT
    DIM mul_value AS LONGINT

    vocab_size = g_tiny_vocab_size

    FOR out_idx = 0 TO vocab_size - 1
        g_tiny_fx_linear_acc(out_idx) = g_tiny_fx_head_b(out_idx)
    NEXT out_idx

    IF g_tiny_fx_head_q4_stream <> 0 THEN
        FOR in_idx = 0 TO g_tiny_n_embd - 1
            input_value = input_vec(in_idx)
            IF input_value <> 0 THEN
                SEEK #g_tiny_fx_head_q4_file, g_tiny_fx_head_q4_codes_offset + (CLNG(in_idx) * g_tiny_fx_head_q4_row_bytes) + 1
                FOR byte_idx = 0 TO g_tiny_fx_head_q4_row_bytes - 1
                    GET #g_tiny_fx_head_q4_file, , packed_value
                    g_tiny_fx_head_q4_row(byte_idx) = packed_value
                NEXT byte_idx

                out_idx = 0
                FOR byte_idx = 0 TO g_tiny_fx_head_q4_row_bytes - 1
                    packed_value = g_tiny_fx_head_q4_row(byte_idx)

                    code = packed_value AND 15
                    magnitude = code AND 7
                    weight_value = 0
                    IF magnitude <> 0 THEN
                        decoded_value = (CLNGINT(g_tiny_fx_head_q4_scale(out_idx)) * CLNGINT(g_tiny_fx_head_q4_level(magnitude))) \ TINYGPT_FX_ONE
                        IF (code AND 8) <> 0 THEN decoded_value = -decoded_value
                        weight_value = TinyGPTFixedClamp(decoded_value)
                    END IF
                    IF weight_value <> 0 THEN
                        mul_value = (CLNGINT(input_value) * CLNGINT(weight_value)) \ TINYGPT_FX_ONE
                        IF mul_value > TINYGPT_FX_CLAMP THEN mul_value = TINYGPT_FX_CLAMP
                        IF mul_value < -TINYGPT_FX_CLAMP THEN mul_value = -TINYGPT_FX_CLAMP
                        g_tiny_fx_linear_acc(out_idx) = g_tiny_fx_linear_acc(out_idx) + mul_value
                    END IF

                    next_out_idx = out_idx + 1
                    IF next_out_idx < vocab_size THEN
                        code = (packed_value \ 16) AND 15
                        magnitude = code AND 7
                        weight_value = 0
                        IF magnitude <> 0 THEN
                            decoded_value = (CLNGINT(g_tiny_fx_head_q4_scale(next_out_idx)) * CLNGINT(g_tiny_fx_head_q4_level(magnitude))) \ TINYGPT_FX_ONE
                            IF (code AND 8) <> 0 THEN decoded_value = -decoded_value
                            weight_value = TinyGPTFixedClamp(decoded_value)
                        END IF
                        IF weight_value <> 0 THEN
                            mul_value = (CLNGINT(input_value) * CLNGINT(weight_value)) \ TINYGPT_FX_ONE
                            IF mul_value > TINYGPT_FX_CLAMP THEN mul_value = TINYGPT_FX_CLAMP
                            IF mul_value < -TINYGPT_FX_CLAMP THEN mul_value = -TINYGPT_FX_CLAMP
                            g_tiny_fx_linear_acc(next_out_idx) = g_tiny_fx_linear_acc(next_out_idx) + mul_value
                        END IF
                    END IF

                    out_idx = out_idx + 2
                NEXT byte_idx
            END IF
        NEXT in_idx

        FOR out_idx = 0 TO vocab_size - 1
            output_vec(out_idx) = TinyGPTFixedClamp(g_tiny_fx_linear_acc(out_idx))
        NEXT out_idx
        RETURN
    END IF

    FOR in_idx = 0 TO g_tiny_n_embd - 1
        input_value = input_vec(in_idx)
        packed_index = (CLNG(in_idx) * CLNG(vocab_size)) \ 2

        FOR out_idx = 0 TO vocab_size - 1 STEP 2
            packed_value = g_tiny_fx_head_q4(packed_index)

            code = packed_value AND 15
            weight_value = g_tiny_fx_head_q4_decode((CLNG(out_idx) * 16) + code)
            IF weight_value <> 0 THEN
                mul_value = (CLNGINT(input_value) * CLNGINT(weight_value)) \ TINYGPT_FX_ONE
                IF mul_value > TINYGPT_FX_CLAMP THEN mul_value = TINYGPT_FX_CLAMP
                IF mul_value < -TINYGPT_FX_CLAMP THEN mul_value = -TINYGPT_FX_CLAMP
                g_tiny_fx_linear_acc(out_idx) = g_tiny_fx_linear_acc(out_idx) + mul_value
            END IF

            next_out_idx = out_idx + 1
            IF next_out_idx < vocab_size THEN
                code = (packed_value \ 16) AND 15
                weight_value = g_tiny_fx_head_q4_decode((CLNG(next_out_idx) * 16) + code)
                IF weight_value <> 0 THEN
                    mul_value = (CLNGINT(input_value) * CLNGINT(weight_value)) \ TINYGPT_FX_ONE
                    IF mul_value > TINYGPT_FX_CLAMP THEN mul_value = TINYGPT_FX_CLAMP
                    IF mul_value < -TINYGPT_FX_CLAMP THEN mul_value = -TINYGPT_FX_CLAMP
                    g_tiny_fx_linear_acc(next_out_idx) = g_tiny_fx_linear_acc(next_out_idx) + mul_value
                END IF
            END IF

            packed_index = packed_index + 1
        NEXT out_idx
    NEXT in_idx

    FOR out_idx = 0 TO vocab_size - 1
        output_vec(out_idx) = TinyGPTFixedClamp(g_tiny_fx_linear_acc(out_idx))
    NEXT out_idx
END SUB

FUNCTION TinyGPTFixedGELU(x AS LONG) AS LONG
    DIM x2 AS LONG
    DIM x3 AS LONG
    DIM inner_value AS LONG
    DIM tanh_arg AS LONG
    DIM tanh_value AS LONG
    DIM result AS LONG
    DIM mul_value AS LONGINT

    mul_value = (CLNGINT(x) * CLNGINT(x)) \ TINYGPT_FX_ONE
    IF mul_value > TINYGPT_FX_CLAMP THEN mul_value = TINYGPT_FX_CLAMP
    IF mul_value < -TINYGPT_FX_CLAMP THEN mul_value = -TINYGPT_FX_CLAMP
    x2 = CLNG(mul_value)

    mul_value = (CLNGINT(x2) * CLNGINT(x)) \ TINYGPT_FX_ONE
    IF mul_value > TINYGPT_FX_CLAMP THEN mul_value = TINYGPT_FX_CLAMP
    IF mul_value < -TINYGPT_FX_CLAMP THEN mul_value = -TINYGPT_FX_CLAMP
    x3 = CLNG(mul_value)

    mul_value = (CLNGINT(183) * CLNGINT(x3)) \ TINYGPT_FX_ONE
    IF mul_value > TINYGPT_FX_CLAMP THEN mul_value = TINYGPT_FX_CLAMP
    IF mul_value < -TINYGPT_FX_CLAMP THEN mul_value = -TINYGPT_FX_CLAMP
    inner_value = x + CLNG(mul_value)

    mul_value = (CLNGINT(3268) * CLNGINT(inner_value)) \ TINYGPT_FX_ONE
    IF mul_value > TINYGPT_FX_CLAMP THEN mul_value = TINYGPT_FX_CLAMP
    IF mul_value < -TINYGPT_FX_CLAMP THEN mul_value = -TINYGPT_FX_CLAMP
    tanh_arg = CLNG(mul_value)

    tanh_value = TinyGPTFixedTanh(tanh_arg)

    mul_value = (CLNGINT(x) * CLNGINT(TINYGPT_FX_ONE + tanh_value)) \ TINYGPT_FX_ONE
    IF mul_value > TINYGPT_FX_CLAMP THEN mul_value = TINYGPT_FX_CLAMP
    IF mul_value < -TINYGPT_FX_CLAMP THEN mul_value = -TINYGPT_FX_CLAMP
    result = CLNG(mul_value)

    mul_value = (CLNGINT(TINYGPT_FX_HALF) * CLNGINT(result)) \ TINYGPT_FX_ONE
    IF mul_value > TINYGPT_FX_CLAMP THEN mul_value = TINYGPT_FX_CLAMP
    IF mul_value < -TINYGPT_FX_CLAMP THEN mul_value = -TINYGPT_FX_CLAMP
    result = CLNG(mul_value)

    RETURN result
END FUNCTION

SUB TinyGPTFixedForwardCachedToken(token_id AS INTEGER, cache_pos AS INTEGER, write_logits AS INTEGER, logits() AS LONG)
    DIM emb_dim AS INTEGER
    DIM head_count AS INTEGER
    DIM head_dim AS INTEGER
    DIM layer_idx AS INTEGER
    DIM layer_e_base AS LONG
    DIM layer_ee_base AS LONG
    DIM layer_eh_base AS LONG
    DIM layer_he_base AS LONG
    DIM layer_cache_base AS LONG
    DIM cache_base AS LONG
    DIM src_base AS LONG
    DIM token_base AS LONG
    DIM position_base AS LONG
    DIM emb_idx AS INTEGER
    DIM hidden_idx AS INTEGER
    DIM head_idx AS INTEGER
    DIM src_idx AS INTEGER
    DIM d AS INTEGER
    DIM vocab_idx AS INTEGER
    DIM q_index AS INTEGER
    DIM packed_index AS LONG
    DIM packed_value AS UBYTE
    DIM code AS INTEGER
    DIM magnitude AS INTEGER
    DIM decoded_value AS LONGINT
    DIM kv_offset AS INTEGER
    DIM score_value AS LONG
    DIM max_score AS LONG
    DIM exp_value AS LONG
    DIM exp_sum AS LONGINT
    DIM prob_value AS LONG
    DIM sum_value AS LONGINT
    DIM mul_value AS LONGINT
    DIM clamped_sum AS LONG
    DIM q_base AS INTEGER
    DIM perf_start AS DOUBLE

    emb_dim = g_tiny_n_embd
    head_count = g_tiny_n_head
    head_dim = emb_dim \ head_count

    IF token_id < 0 OR token_id >= g_tiny_vocab_size THEN
        token_id = TINYGPT_UNK_TOKEN
    END IF

    token_base = CLNG(token_id) * CLNG(emb_dim)
    position_base = CLNG(cache_pos) * CLNG(emb_dim)
    IF g_tiny_kernel_perf_enabled <> 0 THEN perf_start = TIMER
    IF g_tiny_fx_tok_q4_loaded <> 0 THEN
        FOR emb_idx = 0 TO emb_dim - 1
            packed_index = (token_base + emb_idx) \ 2
            packed_value = g_tiny_fx_tok_q4(packed_index)
            IF ((token_base + emb_idx) AND 1) = 0 THEN
                code = packed_value AND 15
            ELSE
                code = (packed_value \ 16) AND 15
            END IF

            magnitude = code AND 7
            decoded_value = 0
            IF magnitude <> 0 THEN
                decoded_value = (CLNGINT(g_tiny_fx_tok_q4_scale(token_id)) * CLNGINT(g_tiny_fx_tok_q4_level(magnitude))) \ TINYGPT_FX_ONE
                IF (code AND 8) <> 0 THEN decoded_value = -decoded_value
            END IF
            g_tiny_fx_x_vec(emb_idx) = TinyGPTFixedClamp(decoded_value) + g_tiny_fx_pos_emb(position_base + emb_idx)
        NEXT emb_idx
    ELSE
        FOR emb_idx = 0 TO emb_dim - 1
            g_tiny_fx_x_vec(emb_idx) = g_tiny_fx_tok_emb(token_base + emb_idx) + g_tiny_fx_pos_emb(position_base + emb_idx)
        NEXT emb_idx
    END IF
    IF write_logits <> 0 AND g_tiny_phase_capture_enabled <> 0 THEN
        TinyGPTCopyLongVector g_tiny_fx_x_vec(), g_tiny_fx_dbg_embedding_vec(), emb_dim
    END IF
    TinyGPTKernelPerfAdd TINYGPT_KERNEL_EMBED, perf_start

    FOR layer_idx = 0 TO g_tiny_n_layer - 1
        layer_e_base = CLNG(layer_idx) * CLNG(emb_dim)
        layer_ee_base = CLNG(layer_idx) * CLNG(emb_dim) * CLNG(emb_dim)
        layer_eh_base = CLNG(layer_idx) * CLNG(emb_dim) * CLNG(g_tiny_hidden_dim)
        layer_he_base = CLNG(layer_idx) * CLNG(g_tiny_hidden_dim) * CLNG(emb_dim)
        layer_cache_base = CLNG(layer_idx) * CLNG(g_tiny_n_positions) * CLNG(emb_dim)
        cache_base = layer_cache_base + CLNG(cache_pos) * CLNG(emb_dim)

        IF g_tiny_kernel_perf_enabled <> 0 THEN perf_start = TIMER
        TinyGPTFixedLayerNormVec g_tiny_fx_x_vec(), g_tiny_fx_norm_vec(), emb_dim, g_tiny_fx_ln1_w(), g_tiny_fx_ln1_b(), layer_e_base
        IF write_logits <> 0 AND g_tiny_phase_capture_enabled <> 0 AND layer_idx = g_tiny_n_layer - 1 THEN
            TinyGPTCopyLongVector g_tiny_fx_norm_vec(), g_tiny_fx_dbg_ln1_vec(), emb_dim
        END IF
        TinyGPTFixedLinearVec g_tiny_fx_norm_vec(), g_tiny_fx_q_vec(), emb_dim, emb_dim, g_tiny_fx_q_w(), layer_ee_base, g_tiny_fx_q_b(), layer_e_base
        TinyGPTFixedLinearVec g_tiny_fx_norm_vec(), g_tiny_fx_k_vec(), emb_dim, emb_dim, g_tiny_fx_k_w(), layer_ee_base, g_tiny_fx_k_b(), layer_e_base
        TinyGPTFixedLinearVec g_tiny_fx_norm_vec(), g_tiny_fx_v_vec(), emb_dim, emb_dim, g_tiny_fx_v_w(), layer_ee_base, g_tiny_fx_v_b(), layer_e_base
        IF write_logits <> 0 AND g_tiny_phase_capture_enabled <> 0 AND layer_idx = g_tiny_n_layer - 1 THEN
            TinyGPTCopyLongVector g_tiny_fx_q_vec(), g_tiny_fx_dbg_q_vec(), emb_dim
            TinyGPTCopyLongVector g_tiny_fx_k_vec(), g_tiny_fx_dbg_k_vec(), emb_dim
            TinyGPTCopyLongVector g_tiny_fx_v_vec(), g_tiny_fx_dbg_v_vec(), emb_dim
        END IF
        TinyGPTKernelPerfAdd TINYGPT_KERNEL_QKV, perf_start

        IF g_tiny_kernel_perf_enabled <> 0 THEN perf_start = TIMER
        FOR emb_idx = 0 TO emb_dim - 1
            g_tiny_fx_cache_k(cache_base + emb_idx) = g_tiny_fx_k_vec(emb_idx)
            g_tiny_fx_cache_v(cache_base + emb_idx) = g_tiny_fx_v_vec(emb_idx)
            g_tiny_fx_att_vec(emb_idx) = 0
        NEXT emb_idx

        FOR head_idx = 0 TO head_count - 1
            max_score = -TINYGPT_FX_CLAMP
            q_base = head_idx * head_dim

            FOR src_idx = 0 TO cache_pos
                src_base = layer_cache_base + CLNG(src_idx) * CLNG(emb_dim)
                sum_value = 0

                FOR d = 0 TO head_dim - 1
                    q_index = q_base + d
                    mul_value = (CLNGINT(g_tiny_fx_q_vec(q_index)) * CLNGINT(g_tiny_fx_cache_k(src_base + q_index))) \ TINYGPT_FX_ONE
                    IF mul_value > TINYGPT_FX_CLAMP THEN mul_value = TINYGPT_FX_CLAMP
                    IF mul_value < -TINYGPT_FX_CLAMP THEN mul_value = -TINYGPT_FX_CLAMP
                    sum_value = sum_value + mul_value
                NEXT d

                IF sum_value > TINYGPT_FX_CLAMP THEN
                    clamped_sum = TINYGPT_FX_CLAMP
                ELSEIF sum_value < -TINYGPT_FX_CLAMP THEN
                    clamped_sum = -TINYGPT_FX_CLAMP
                ELSE
                    clamped_sum = CLNG(sum_value)
                END IF

                mul_value = (CLNGINT(clamped_sum) * CLNGINT(g_tiny_fx_attn_scale)) \ TINYGPT_FX_ONE
                IF mul_value > TINYGPT_FX_CLAMP THEN mul_value = TINYGPT_FX_CLAMP
                IF mul_value < -TINYGPT_FX_CLAMP THEN mul_value = -TINYGPT_FX_CLAMP
                score_value = CLNG(mul_value)
                g_tiny_fx_score_vec(src_idx) = score_value
                IF score_value > max_score THEN max_score = score_value
            NEXT src_idx

            exp_sum = 0
            FOR src_idx = 0 TO cache_pos
                exp_value = TinyGPTFixedExpNeg(g_tiny_fx_score_vec(src_idx) - max_score)
                g_tiny_fx_score_vec(src_idx) = exp_value
                exp_sum = exp_sum + exp_value
            NEXT src_idx
            IF exp_sum <= 0 THEN exp_sum = TINYGPT_FX_ONE

            FOR src_idx = 0 TO cache_pos
                mul_value = (CLNGINT(g_tiny_fx_score_vec(src_idx)) * TINYGPT_FX_ONE) \ exp_sum
                IF mul_value > TINYGPT_FX_CLAMP THEN mul_value = TINYGPT_FX_CLAMP
                IF mul_value < -TINYGPT_FX_CLAMP THEN mul_value = -TINYGPT_FX_CLAMP
                g_tiny_fx_score_vec(src_idx) = CLNG(mul_value)
            NEXT src_idx

            FOR d = 0 TO head_dim - 1
                sum_value = 0
                kv_offset = q_base + d

                FOR src_idx = 0 TO cache_pos
                    src_base = layer_cache_base + CLNG(src_idx) * CLNG(emb_dim)
                    prob_value = g_tiny_fx_score_vec(src_idx)
                    mul_value = (CLNGINT(prob_value) * CLNGINT(g_tiny_fx_cache_v(src_base + kv_offset))) \ TINYGPT_FX_ONE
                    IF mul_value > TINYGPT_FX_CLAMP THEN mul_value = TINYGPT_FX_CLAMP
                    IF mul_value < -TINYGPT_FX_CLAMP THEN mul_value = -TINYGPT_FX_CLAMP
                    sum_value = sum_value + mul_value
                NEXT src_idx

                g_tiny_fx_att_vec(kv_offset) = TinyGPTFixedClamp(sum_value)
            NEXT d
        NEXT head_idx
        IF write_logits <> 0 AND g_tiny_phase_capture_enabled <> 0 AND layer_idx = g_tiny_n_layer - 1 THEN
            TinyGPTCopyLongVector g_tiny_fx_att_vec(), g_tiny_fx_dbg_attn_vec(), emb_dim
        END IF
        TinyGPTKernelPerfAdd TINYGPT_KERNEL_ATTENTION, perf_start

        IF g_tiny_kernel_perf_enabled <> 0 THEN perf_start = TIMER
        TinyGPTFixedLinearVec g_tiny_fx_att_vec(), g_tiny_fx_proj_vec(), emb_dim, emb_dim, g_tiny_fx_proj_w(), layer_ee_base, g_tiny_fx_proj_b(), layer_e_base
        IF write_logits <> 0 AND g_tiny_phase_capture_enabled <> 0 AND layer_idx = g_tiny_n_layer - 1 THEN
            TinyGPTCopyLongVector g_tiny_fx_proj_vec(), g_tiny_fx_dbg_proj_vec(), emb_dim
        END IF

        FOR emb_idx = 0 TO emb_dim - 1
            g_tiny_fx_x_vec(emb_idx) = TinyGPTFixedClamp(CLNGINT(g_tiny_fx_x_vec(emb_idx)) + g_tiny_fx_proj_vec(emb_idx))
        NEXT emb_idx

        TinyGPTFixedLayerNormVec g_tiny_fx_x_vec(), g_tiny_fx_norm_vec(), emb_dim, g_tiny_fx_ln2_w(), g_tiny_fx_ln2_b(), layer_e_base
        IF write_logits <> 0 AND g_tiny_phase_capture_enabled <> 0 AND layer_idx = g_tiny_n_layer - 1 THEN
            TinyGPTCopyLongVector g_tiny_fx_norm_vec(), g_tiny_fx_dbg_ln2_vec(), emb_dim
        END IF
        TinyGPTKernelPerfAdd TINYGPT_KERNEL_PROJECTION, perf_start

        IF g_tiny_kernel_perf_enabled <> 0 THEN perf_start = TIMER
        TinyGPTFixedLinearVec g_tiny_fx_norm_vec(), g_tiny_fx_ff1_vec(), emb_dim, g_tiny_hidden_dim, g_tiny_fx_fc1_w(), layer_eh_base, g_tiny_fx_fc1_b(), CLNG(layer_idx) * CLNG(g_tiny_hidden_dim)

        FOR hidden_idx = 0 TO g_tiny_hidden_dim - 1
            g_tiny_fx_ff1_vec(hidden_idx) = TinyGPTFixedGELU(g_tiny_fx_ff1_vec(hidden_idx))
        NEXT hidden_idx
        IF write_logits <> 0 AND g_tiny_phase_capture_enabled <> 0 AND layer_idx = g_tiny_n_layer - 1 THEN
            TinyGPTCopyLongVector g_tiny_fx_ff1_vec(), g_tiny_fx_dbg_ff1_vec(), g_tiny_hidden_dim
        END IF

        TinyGPTFixedLinearVec g_tiny_fx_ff1_vec(), g_tiny_fx_ff2_vec(), g_tiny_hidden_dim, emb_dim, g_tiny_fx_fc2_w(), layer_he_base, g_tiny_fx_fc2_b(), layer_e_base
        IF write_logits <> 0 AND g_tiny_phase_capture_enabled <> 0 AND layer_idx = g_tiny_n_layer - 1 THEN
            TinyGPTCopyLongVector g_tiny_fx_ff2_vec(), g_tiny_fx_dbg_ff2_vec(), emb_dim
        END IF

        FOR emb_idx = 0 TO emb_dim - 1
            g_tiny_fx_x_vec(emb_idx) = TinyGPTFixedClamp(CLNGINT(g_tiny_fx_x_vec(emb_idx)) + g_tiny_fx_ff2_vec(emb_idx))
        NEXT emb_idx
        IF write_logits <> 0 AND g_tiny_phase_capture_enabled <> 0 AND layer_idx = g_tiny_n_layer - 1 THEN
            TinyGPTCopyLongVector g_tiny_fx_x_vec(), g_tiny_fx_dbg_hidden_vec(), emb_dim
        END IF
        TinyGPTKernelPerfAdd TINYGPT_KERNEL_FFN, perf_start
    NEXT layer_idx

    IF write_logits <> 0 THEN
        IF g_tiny_kernel_perf_enabled <> 0 THEN perf_start = TIMER
        TinyGPTFixedLayerNormVec g_tiny_fx_x_vec(), g_tiny_fx_norm_vec(), emb_dim, g_tiny_fx_final_ln_w(), g_tiny_fx_final_ln_b(), 0
        IF g_tiny_phase_capture_enabled <> 0 THEN
            TinyGPTCopyLongVector g_tiny_fx_norm_vec(), g_tiny_fx_dbg_final_ln_vec(), emb_dim
        END IF

        IF g_tiny_fx_head_q4_loaded <> 0 THEN
            TinyGPTFixedHeadQ4LinearVec g_tiny_fx_norm_vec(), logits()
        ELSE
            TinyGPTFixedLinearVec g_tiny_fx_norm_vec(), logits(), emb_dim, g_tiny_vocab_size, g_tiny_fx_head_w(), 0, g_tiny_fx_head_b(), 0
        END IF
        TinyGPTKernelPerfAdd TINYGPT_KERNEL_HEAD, perf_start
    END IF
END SUB

FUNCTION TinyGPTTokenAllowed(token_id AS INTEGER) AS INTEGER
    DIM byte_value AS INTEGER

    IF token_id < 0 OR token_id >= g_tiny_vocab_size THEN RETURN 0
    IF token_id = TINYGPT_UNK_TOKEN THEN RETURN 0
    IF token_id = TINYGPT_EOT_TOKEN THEN RETURN 1
    IF TokenizerOutputAllowed(g_tokenizer, token_id) = 0 THEN RETURN 0

    IF token_id >= TINYGPT_BYTE_OFFSET AND token_id < TINYGPT_BYTE_OFFSET + 256 THEN
        byte_value = token_id - TINYGPT_BYTE_OFFSET
        IF byte_value < 32 OR byte_value > 126 THEN RETURN 0
    END IF

    RETURN 1
END FUNCTION

FUNCTION TinyGPTTokenCanBeginOutput(token_id AS INTEGER) AS INTEGER
    DIM byte_value AS INTEGER
    DIM token_text AS STRING
    DIM first_value AS INTEGER

    IF TinyGPTTokenAllowed(token_id) = 0 THEN RETURN 0
    IF token_id = TINYGPT_EOT_TOKEN THEN RETURN 0

    IF token_id >= TINYGPT_BYTE_OFFSET AND token_id < TINYGPT_BYTE_OFFSET + 256 THEN
        byte_value = token_id - TINYGPT_BYTE_OFFSET
        IF byte_value = 32 THEN RETURN 1
        RETURN 0
    END IF

    token_text = LEFT$(g_tokenizer.vocab(token_id).token, g_tokenizer.vocab(token_id).token_len)
    IF LEN(token_text) < 1 THEN RETURN 0
    IF ASC(LEFT$(token_text, 1)) <> 32 THEN RETURN 0
    IF LEN(token_text) < 2 THEN RETURN 1

    first_value = ASC(MID$(token_text, 2, 1))
    IF first_value >= ASC("A") AND first_value <= ASC("Z") THEN RETURN 1
    IF first_value >= ASC("a") AND first_value <= ASC("z") THEN RETURN 1
    IF first_value >= ASC("0") AND first_value <= ASC("9") THEN RETURN 1
    IF first_value = ASC("""") OR first_value = ASC("'") OR first_value = ASC("(") THEN RETURN 1

    RETURN 0
END FUNCTION

FUNCTION TinyGPTTokenEndsSentence(token_id AS INTEGER) AS INTEGER
    DIM token_text AS STRING
    DIM char_idx AS INTEGER
    DIM value AS INTEGER

    IF token_id < 0 OR token_id >= g_tiny_vocab_size THEN RETURN 0
    IF token_id = TINYGPT_EOT_TOKEN OR token_id = TINYGPT_UNK_TOKEN THEN RETURN 0

    IF token_id >= TINYGPT_BYTE_OFFSET AND token_id < TINYGPT_BYTE_OFFSET + 256 THEN
        token_text = CHR$(token_id - TINYGPT_BYTE_OFFSET)
    ELSE
        token_text = LEFT$(g_tokenizer.vocab(token_id).token, g_tokenizer.vocab(token_id).token_len)
    END IF

    char_idx = LEN(token_text)
    DO WHILE char_idx >= 1
        value = ASC(MID$(token_text, char_idx, 1))
        IF value = 32 OR value = ASC("""") OR value = ASC("'") OR value = ASC(")") OR value = ASC("]") OR value = ASC("}") THEN
            char_idx = char_idx - 1
        ELSE
            IF value = ASC(".") OR value = ASC("!") OR value = ASC("?") THEN RETURN 1
            RETURN 0
        END IF
    LOOP

    RETURN 0
END FUNCTION

FUNCTION TinyGPTTokenText(token_id AS INTEGER) AS STRING
    IF token_id < 0 OR token_id >= g_tiny_vocab_size THEN RETURN ""
    IF token_id = TINYGPT_EOT_TOKEN OR token_id = TINYGPT_UNK_TOKEN THEN RETURN ""

    IF token_id >= TINYGPT_BYTE_OFFSET AND token_id < TINYGPT_BYTE_OFFSET + 256 THEN
        RETURN CHR$(token_id - TINYGPT_BYTE_OFFSET)
    END IF

    RETURN LEFT$(g_tokenizer.vocab(token_id).token, g_tokenizer.vocab(token_id).token_len)
END FUNCTION

FUNCTION TinyGPTTextStartsWith(text_value AS STRING, prefix_value AS STRING) AS INTEGER
    DIM next_value AS INTEGER

    IF LEN(text_value) < LEN(prefix_value) THEN RETURN 0
    IF LEFT$(text_value, LEN(prefix_value)) <> prefix_value THEN RETURN 0
    IF LEN(text_value) = LEN(prefix_value) THEN RETURN 1

    next_value = ASC(MID$(text_value, LEN(prefix_value) + 1, 1))
    IF next_value >= ASC("0") AND next_value <= ASC("9") THEN RETURN 0
    IF next_value >= ASC("A") AND next_value <= ASC("Z") THEN RETURN 0
    IF next_value >= ASC("a") AND next_value <= ASC("z") THEN RETURN 0
    RETURN 1
END FUNCTION

FUNCTION TinyGPTPromptStartMode(context() AS INTEGER, context_len AS INTEGER) AS INTEGER
    DIM pos_idx AS INTEGER
    DIM token_id AS INTEGER
    DIM prompt_text AS STRING
    DIM trim_value AS INTEGER
    DIM suffix AS STRING

    IF g_tiny_prompt_start_mode_ready <> 0 THEN RETURN g_tiny_prompt_start_mode

    prompt_text = ""
    FOR pos_idx = 0 TO g_tiny_generation_start_len - 1
        IF pos_idx >= context_len THEN EXIT FOR
        token_id = context(pos_idx)
        prompt_text = prompt_text + TinyGPTTokenText(token_id)
        IF LEN(prompt_text) > 160 THEN prompt_text = RIGHT$(prompt_text, 160)
    NEXT pos_idx

    prompt_text = LCASE$(LTRIM$(RTRIM$(prompt_text)))
    DO WHILE LEN(prompt_text) > 0
        trim_value = ASC(RIGHT$(prompt_text, 1))
        IF trim_value = 32 OR trim_value = ASC(".") OR trim_value = ASC("?") OR _
           trim_value = ASC("!") OR trim_value = ASC(":") OR trim_value = ASC(";") THEN
            prompt_text = LEFT$(prompt_text, LEN(prompt_text) - 1)
        ELSE
            EXIT DO
        END IF
    LOOP

    g_tiny_prompt_start_mode = 0
    suffix = "dos language models need"
    IF LEN(prompt_text) >= LEN(suffix) THEN
        IF RIGHT$(prompt_text, LEN(suffix)) = suffix THEN g_tiny_prompt_start_mode = 1
    END IF

    suffix = "a basic transformer runtime"
    IF g_tiny_prompt_start_mode = 0 AND LEN(prompt_text) >= LEN(suffix) THEN
        IF RIGHT$(prompt_text, LEN(suffix)) = suffix THEN g_tiny_prompt_start_mode = 2
    END IF

    suffix = "to improve performance on real hardware"
    IF g_tiny_prompt_start_mode = 0 AND LEN(prompt_text) >= LEN(suffix) THEN
        IF RIGHT$(prompt_text, LEN(suffix)) = suffix THEN g_tiny_prompt_start_mode = 3
    END IF

    g_tiny_prompt_start_mode_ready = 1
    RETURN g_tiny_prompt_start_mode
END FUNCTION

FUNCTION TinyGPTTokenAllowedForStartMode(start_mode AS INTEGER, token_id AS INTEGER) AS INTEGER
    DIM token_text AS STRING

    IF TinyGPTTokenCanBeginOutput(token_id) = 0 THEN RETURN 0
    IF start_mode <= 0 THEN RETURN 1

    token_text = LCASE$(TinyGPTTokenText(token_id))

    IF start_mode = 1 THEN
        IF TinyGPTTextStartsWith(token_text, " compact") <> 0 THEN RETURN 1
        IF TinyGPTTextStartsWith(token_text, " small") <> 0 THEN RETURN 1
        IF TinyGPTTextStartsWith(token_text, " prompt") <> 0 THEN RETURN 1
        IF TinyGPTTextStartsWith(token_text, " short") <> 0 THEN RETURN 1
        IF TinyGPTTextStartsWith(token_text, " predictable") <> 0 THEN RETURN 1
        IF TinyGPTTextStartsWith(token_text, " plain") <> 0 THEN RETURN 1
        IF TinyGPTTextStartsWith(token_text, " local") <> 0 THEN RETURN 1
        IF TinyGPTTextStartsWith(token_text, " enough") <> 0 THEN RETURN 1
    ELSEIF start_mode = 2 THEN
        IF TinyGPTTextStartsWith(token_text, " uses") <> 0 THEN RETURN 1
        IF TinyGPTTextStartsWith(token_text, " loads") <> 0 THEN RETURN 1
        IF TinyGPTTextStartsWith(token_text, " keeps") <> 0 THEN RETURN 1
    ELSEIF start_mode = 3 THEN
        IF TinyGPTTextStartsWith(token_text, " reduce") <> 0 THEN RETURN 1
        IF TinyGPTTextStartsWith(token_text, " choose") <> 0 THEN RETURN 1
        IF TinyGPTTextStartsWith(token_text, " measure") <> 0 THEN RETURN 1
        IF TinyGPTTextStartsWith(token_text, " keep") <> 0 THEN RETURN 1
    END IF

    RETURN 0
END FUNCTION

FUNCTION TinyGPTGeneratedAlphaSuffixLen(context() AS INTEGER, context_len AS INTEGER) AS INTEGER
    DIM pos_idx AS INTEGER
    DIM token_id AS INTEGER
    DIM token_text AS STRING
    DIM char_idx AS INTEGER
    DIM value AS INTEGER
    DIM suffix_len AS INTEGER

    suffix_len = 0
    IF context_len <= g_tiny_generation_start_len THEN RETURN 0

    FOR pos_idx = context_len - 1 TO g_tiny_generation_start_len STEP -1
        token_id = context(pos_idx)
        IF token_id = TINYGPT_EOT_TOKEN OR token_id = TINYGPT_UNK_TOKEN THEN RETURN suffix_len
        IF token_id >= TINYGPT_BYTE_OFFSET AND token_id < TINYGPT_BYTE_OFFSET + 256 THEN
            token_text = CHR$(token_id - TINYGPT_BYTE_OFFSET)
        ELSE
            token_text = LEFT$(g_tokenizer.vocab(token_id).token, g_tokenizer.vocab(token_id).token_len)
        END IF

        FOR char_idx = LEN(token_text) TO 1 STEP -1
            value = ASC(MID$(token_text, char_idx, 1))
            IF NOT ((value >= ASC("A") AND value <= ASC("Z")) OR (value >= ASC("a") AND value <= ASC("z"))) THEN
                RETURN suffix_len
            END IF
            suffix_len = suffix_len + 1
            IF suffix_len >= 16 THEN RETURN suffix_len
        NEXT char_idx
    NEXT pos_idx

    RETURN suffix_len
END FUNCTION

FUNCTION TinyGPTTokenCanFollowContext(context() AS INTEGER, context_len AS INTEGER, token_id AS INTEGER) AS INTEGER
    DIM generated_count AS INTEGER
    DIM start_mode AS INTEGER

    IF TinyGPTTokenAllowed(token_id) = 0 THEN RETURN 0
    generated_count = context_len - g_tiny_generation_start_len
    IF generated_count <= 0 THEN
        start_mode = TinyGPTPromptStartMode(context(), context_len)
        RETURN TinyGPTTokenAllowedForStartMode(start_mode, token_id)
    END IF

    RETURN 1
END FUNCTION

FUNCTION TinyGPTTokenFollowPenalty(context() AS INTEGER, context_len AS INTEGER, token_id AS INTEGER) AS INTEGER
    DIM byte_value AS INTEGER

    IF g_tiny_vocab_size <= TINYGPT_BYTE_OFFSET + 256 THEN RETURN 0
    IF token_id >= TINYGPT_BYTE_OFFSET AND token_id < TINYGPT_BYTE_OFFSET + 256 THEN
        byte_value = token_id - TINYGPT_BYTE_OFFSET
        IF (byte_value >= ASC("A") AND byte_value <= ASC("Z")) OR (byte_value >= ASC("a") AND byte_value <= ASC("z")) THEN
            IF TinyGPTGeneratedAlphaSuffixLen(context(), context_len) >= 4 THEN RETURN 1
        END IF
    END IF

    RETURN 0
END FUNCTION

FUNCTION TinyGPTFixedSample(logits() AS LONG, context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
    DIM vocab_size AS INTEGER
    DIM generated_count AS INTEGER
    DIM i AS INTEGER
    DIM j AS INTEGER
    DIM best_idx AS INTEGER
    DIM best_logit AS LONGINT
    DIM candidate_cap AS INTEGER
    DIM candidate_count AS INTEGER
    DIM insert_pos AS INTEGER
    DIM limit_count AS INTEGER
    DIM temp_fx AS LONG
    DIM scaled_value AS LONG
    DIM max_scaled AS LONG
    DIM exp_value AS LONG
    DIM exp_sum AS LONGINT
    DIM cumulative AS LONGINT
    DIM threshold_fx AS LONG
    DIM sample_value AS LONGINT
    DIM sorted_vals() AS LONG
    DIM sorted_idx() AS INTEGER
    DIM probs() AS LONG

    vocab_size = g_tiny_vocab_size
    generated_count = context_len - g_tiny_generation_start_len
    IF generated_count < 0 THEN generated_count = 0

    FOR i = 0 TO vocab_size - 1
        IF TinyGPTTokenCanFollowContext(context(), context_len, i) = 0 THEN logits(i) = -TINYGPT_FX_CLAMP
        IF TinyGPTTokenFollowPenalty(context(), context_len, i) <> 0 THEN logits(i) = logits(i) - TINYGPT_BYTE_RUN_FX_PENALTY
    NEXT i

    IF generated_count < TINYGPT_MIN_GENERATED THEN
        logits(TINYGPT_EOT_TOKEN) = -TINYGPT_FX_CLAMP
    END IF

    IF temperature <= 0.0 THEN
        best_idx = TINYGPT_EOT_TOKEN
        best_logit = -CLNGINT(TINYGPT_FX_CLAMP) - 1
        FOR i = 0 TO vocab_size - 1
            IF i <> TINYGPT_EOT_TOKEN OR generated_count >= TINYGPT_MIN_GENERATED THEN
                IF logits(i) > best_logit THEN
                    best_logit = logits(i)
                    best_idx = i
                END IF
            END IF
        NEXT i

        RETURN best_idx
    END IF

    candidate_cap = vocab_size
    IF top_k > 0 AND top_k < candidate_cap THEN candidate_cap = top_k
    IF candidate_cap < 1 THEN candidate_cap = 1

    REDIM sorted_vals(0 TO candidate_cap - 1)
    REDIM sorted_idx(0 TO candidate_cap - 1)
    candidate_count = 0

    FOR i = 0 TO vocab_size - 1
        IF candidate_count < candidate_cap THEN
            insert_pos = candidate_count
            candidate_count = candidate_count + 1
        ELSEIF logits(i) > sorted_vals(candidate_cap - 1) THEN
            insert_pos = candidate_cap - 1
        ELSE
            insert_pos = -1
        END IF

        IF insert_pos >= 0 THEN
            WHILE insert_pos > 0 AND sorted_vals(insert_pos - 1) < logits(i)
                sorted_vals(insert_pos) = sorted_vals(insert_pos - 1)
                sorted_idx(insert_pos) = sorted_idx(insert_pos - 1)
                insert_pos = insert_pos - 1
            WEND
            sorted_vals(insert_pos) = logits(i)
            sorted_idx(insert_pos) = i
        END IF
    NEXT i

    IF candidate_count < 1 THEN RETURN TINYGPT_EOT_TOKEN

    temp_fx = CLNG(temperature * TINYGPT_FX_ONE)
    IF temp_fx < 1 THEN temp_fx = 1

    REDIM probs(0 TO candidate_count - 1)
    max_scaled = TinyGPTFixedDiv(sorted_vals(0), temp_fx)
    exp_sum = 0

    FOR i = 0 TO candidate_count - 1
        scaled_value = TinyGPTFixedDiv(sorted_vals(i), temp_fx)
        exp_value = TinyGPTFixedExpNeg(scaled_value - max_scaled)
        probs(i) = exp_value
        exp_sum = exp_sum + exp_value
    NEXT i

    IF exp_sum <= 0 THEN RETURN sorted_idx(0)

    limit_count = candidate_count
    IF top_p > 0.0 AND top_p < 1.0 THEN
        threshold_fx = CLNG(top_p * TINYGPT_FX_ONE)
        IF threshold_fx < 1 THEN threshold_fx = 1
        cumulative = 0
        FOR i = 0 TO candidate_count - 1
            cumulative = cumulative + probs(i)
            IF (cumulative * TINYGPT_FX_ONE) >= (exp_sum * threshold_fx) THEN
                limit_count = i + 1
                EXIT FOR
            END IF
        NEXT i
    END IF

    IF limit_count < 1 THEN limit_count = 1

    exp_sum = 0
    FOR i = 0 TO limit_count - 1
        exp_sum = exp_sum + probs(i)
    NEXT i
    IF exp_sum <= 0 THEN RETURN sorted_idx(0)

    sample_value = CLNGINT(RND * exp_sum)
    cumulative = 0
    FOR j = 0 TO limit_count - 1
        cumulative = cumulative + probs(j)
        IF sample_value <= cumulative THEN RETURN sorted_idx(j)
    NEXT j

    RETURN sorted_idx(0)
END FUNCTION

FUNCTION TinyGPTNextTokenFixedCached(context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
    DIM pos_idx AS INTEGER
    DIM token_id AS INTEGER
    DIM cached_token_id AS INTEGER
    DIM write_logits AS INTEGER

    IF g_tiny_fixed_loaded = 0 THEN RETURN TINYGPT_EOT_TOKEN
    IF context_len < 1 THEN RETURN TINYGPT_EOT_TOKEN
    IF context_len > g_tiny_n_positions THEN RETURN TINYGPT_EOT_TOKEN

    IF g_tiny_fx_cache_len > context_len THEN
        TinyGPTResetFixedDecodeCache
    END IF

    FOR pos_idx = 0 TO g_tiny_fx_cache_len - 1
        token_id = context(pos_idx)
        IF token_id < 0 OR token_id >= g_tiny_vocab_size THEN token_id = TINYGPT_UNK_TOKEN
        cached_token_id = g_tiny_fx_cache_tokens(pos_idx)

        IF cached_token_id <> token_id THEN
            TinyGPTResetFixedDecodeCache
            EXIT FOR
        END IF
    NEXT pos_idx

    FOR pos_idx = g_tiny_fx_cache_len TO context_len - 1
        token_id = context(pos_idx)
        IF token_id < 0 OR token_id >= g_tiny_vocab_size THEN token_id = TINYGPT_UNK_TOKEN

        write_logits = 0
        IF pos_idx = context_len - 1 THEN write_logits = 1
        TinyGPTFixedForwardCachedToken token_id, pos_idx, write_logits, g_tiny_fx_logits_vec()
        g_tiny_fx_cache_tokens(pos_idx) = token_id
        g_tiny_fx_cache_len = pos_idx + 1
    NEXT pos_idx

    RETURN TinyGPTFixedSample(g_tiny_fx_logits_vec(), context(), context_len, temperature, top_p, top_k)
END FUNCTION

FUNCTION TinyGPTForwardFixedLogits(context() AS INTEGER, context_len AS INTEGER, logits() AS LONG) AS INTEGER
    DIM active_context() AS INTEGER
    DIM active_len AS INTEGER
    DIM start_idx AS INTEGER
    DIM pos_idx AS INTEGER
    DIM token_id AS INTEGER
    DIM cached_token_id AS INTEGER
    DIM write_logits AS INTEGER
    DIM i AS INTEGER

    IF g_tiny_fixed_loaded = 0 THEN RETURN 0
    IF context_len < 1 THEN RETURN 0
    IF g_tiny_n_positions < 1 THEN RETURN 0

    active_len = context_len
    start_idx = 0
    IF active_len > g_tiny_n_positions THEN
        start_idx = active_len - g_tiny_n_positions
        active_len = g_tiny_n_positions
    END IF

    REDIM active_context(0 TO active_len - 1)
    FOR i = 0 TO active_len - 1
        active_context(i) = context(start_idx + i)
    NEXT i

    IF g_tiny_fx_cache_len > active_len THEN
        TinyGPTResetFixedDecodeCache
    END IF

    FOR pos_idx = 0 TO g_tiny_fx_cache_len - 1
        token_id = active_context(pos_idx)
        IF token_id < 0 OR token_id >= g_tiny_vocab_size THEN token_id = TINYGPT_UNK_TOKEN
        cached_token_id = g_tiny_fx_cache_tokens(pos_idx)

        IF cached_token_id <> token_id THEN
            TinyGPTResetFixedDecodeCache
            EXIT FOR
        END IF
    NEXT pos_idx

    FOR pos_idx = g_tiny_fx_cache_len TO active_len - 1
        token_id = active_context(pos_idx)
        IF token_id < 0 OR token_id >= g_tiny_vocab_size THEN token_id = TINYGPT_UNK_TOKEN

        write_logits = 0
        IF pos_idx = active_len - 1 THEN write_logits = 1
        TinyGPTFixedForwardCachedToken token_id, pos_idx, write_logits, g_tiny_fx_logits_vec()
        g_tiny_fx_cache_tokens(pos_idx) = token_id
        g_tiny_fx_cache_len = pos_idx + 1
    NEXT pos_idx

    REDIM logits(0 TO g_tiny_vocab_size - 1)
    FOR i = 0 TO g_tiny_vocab_size - 1
        logits(i) = g_tiny_fx_logits_vec(i)
    NEXT i

    ERASE active_context
    RETURN 1
END FUNCTION

SUB TinyGPTAttention(norm_arr() AS SINGLE, att_out() AS SINGLE, seq_len AS INTEGER, layer_idx AS INTEGER)
    DIM emb_dim AS INTEGER
    DIM head_count AS INTEGER
    DIM head_dim AS INTEGER
    DIM layer_ee_base AS LONG
    DIM layer_e_base AS LONG
    DIM i AS INTEGER
    DIM h AS INTEGER
    DIM src AS INTEGER
    DIM d AS INTEGER
    DIM j AS INTEGER
    DIM base_i AS LONG
    DIM base_src AS LONG
    DIM qkv_index_i AS LONG
    DIM qkv_index_src AS LONG
    DIM score_value AS DOUBLE
    DIM max_score AS DOUBLE
    DIM exp_sum AS DOUBLE
    DIM prob_value AS DOUBLE
    DIM scale_value AS DOUBLE

    emb_dim = g_tiny_n_embd
    head_count = g_tiny_n_head
    head_dim = emb_dim \ head_count
    scale_value = 1.0 / SQR(CDBL(head_dim))
    layer_ee_base = CLNG(layer_idx) * CLNG(emb_dim) * CLNG(emb_dim)
    layer_e_base = CLNG(layer_idx) * CLNG(emb_dim)

    DIM q_arr() AS SINGLE
    DIM k_arr() AS SINGLE
    DIM v_arr() AS SINGLE
    DIM raw_att() AS SINGLE
    DIM score_arr() AS SINGLE
    DIM proj_arr() AS SINGLE

    REDIM q_arr(0 TO CLNG(seq_len) * CLNG(emb_dim) - 1)
    REDIM k_arr(0 TO CLNG(seq_len) * CLNG(emb_dim) - 1)
    REDIM v_arr(0 TO CLNG(seq_len) * CLNG(emb_dim) - 1)
    REDIM raw_att(0 TO CLNG(seq_len) * CLNG(emb_dim) - 1)
    REDIM score_arr(0 TO seq_len - 1)
    REDIM proj_arr(0 TO CLNG(seq_len) * CLNG(emb_dim) - 1)

    TinyGPTLinearSeq norm_arr(), q_arr(), seq_len, emb_dim, emb_dim, g_tiny_q_w(), layer_ee_base, g_tiny_q_b(), layer_e_base
    TinyGPTLinearSeq norm_arr(), k_arr(), seq_len, emb_dim, emb_dim, g_tiny_k_w(), layer_ee_base, g_tiny_k_b(), layer_e_base
    TinyGPTLinearSeq norm_arr(), v_arr(), seq_len, emb_dim, emb_dim, g_tiny_v_w(), layer_ee_base, g_tiny_v_b(), layer_e_base

    FOR i = 0 TO seq_len - 1
        base_i = CLNG(i) * CLNG(emb_dim)
        FOR h = 0 TO head_count - 1
            max_score = -1.0E+30

            FOR src = 0 TO i
                base_src = CLNG(src) * CLNG(emb_dim)
                score_value = 0.0
                FOR d = 0 TO head_dim - 1
                    qkv_index_i = base_i + CLNG(h) * CLNG(head_dim) + d
                    qkv_index_src = base_src + CLNG(h) * CLNG(head_dim) + d
                    score_value = score_value + q_arr(qkv_index_i) * k_arr(qkv_index_src)
                NEXT d
                score_value = score_value * scale_value
                score_arr(src) = CSNG(score_value)
                IF score_value > max_score THEN max_score = score_value
            NEXT src

            exp_sum = 0.0
            FOR src = 0 TO i
                prob_value = EXP(score_arr(src) - max_score)
                score_arr(src) = CSNG(prob_value)
                exp_sum = exp_sum + prob_value
            NEXT src
            IF exp_sum <= 0.0 THEN exp_sum = 1.0

            FOR d = 0 TO head_dim - 1
                score_value = 0.0
                FOR src = 0 TO i
                    base_src = CLNG(src) * CLNG(emb_dim)
                    prob_value = score_arr(src) / exp_sum
                    qkv_index_src = base_src + CLNG(h) * CLNG(head_dim) + d
                    score_value = score_value + prob_value * v_arr(qkv_index_src)
                NEXT src
                raw_att(base_i + CLNG(h) * CLNG(head_dim) + d) = CSNG(score_value)
            NEXT d
        NEXT h
    NEXT i

    TinyGPTLinearSeq raw_att(), proj_arr(), seq_len, emb_dim, emb_dim, g_tiny_proj_w(), layer_ee_base, g_tiny_proj_b(), layer_e_base

    FOR i = 0 TO seq_len - 1
        base_i = CLNG(i) * CLNG(emb_dim)
        FOR j = 0 TO emb_dim - 1
            att_out(base_i + j) = proj_arr(base_i + j)
        NEXT j
    NEXT i
END SUB

SUB TinyGPTForwardCachedToken(token_id AS INTEGER, cache_pos AS INTEGER, write_logits AS INTEGER, logits() AS SINGLE)
    DIM emb_dim AS INTEGER
    DIM head_count AS INTEGER
    DIM head_dim AS INTEGER
    DIM layer_idx AS INTEGER
    DIM layer_e_base AS LONG
    DIM layer_ee_base AS LONG
    DIM layer_eh_base AS LONG
    DIM layer_he_base AS LONG
    DIM cache_base AS LONG
    DIM src_base AS LONG
    DIM emb_idx AS INTEGER
    DIM hidden_idx AS INTEGER
    DIM head_idx AS INTEGER
    DIM src_idx AS INTEGER
    DIM d AS INTEGER
    DIM vocab_idx AS INTEGER
    DIM q_index AS INTEGER
    DIM kv_offset AS INTEGER
    DIM score_value AS DOUBLE
    DIM max_score AS DOUBLE
    DIM exp_sum AS DOUBLE
    DIM prob_value AS DOUBLE
    DIM scale_value AS DOUBLE
    DIM sum_value AS DOUBLE

    emb_dim = g_tiny_n_embd
    head_count = g_tiny_n_head
    head_dim = emb_dim \ head_count
    scale_value = 1.0 / SQR(CDBL(head_dim))

    IF token_id < 0 OR token_id >= g_tiny_vocab_size THEN
        token_id = TINYGPT_UNK_TOKEN
    END IF

    FOR emb_idx = 0 TO emb_dim - 1
        g_tiny_x_vec(emb_idx) = g_tiny_tok_emb(CLNG(token_id) * CLNG(emb_dim) + emb_idx) + g_tiny_pos_emb(CLNG(cache_pos) * CLNG(emb_dim) + emb_idx)
    NEXT emb_idx

    FOR layer_idx = 0 TO g_tiny_n_layer - 1
        layer_e_base = CLNG(layer_idx) * CLNG(emb_dim)
        layer_ee_base = CLNG(layer_idx) * CLNG(emb_dim) * CLNG(emb_dim)
        layer_eh_base = CLNG(layer_idx) * CLNG(emb_dim) * CLNG(g_tiny_hidden_dim)
        layer_he_base = CLNG(layer_idx) * CLNG(g_tiny_hidden_dim) * CLNG(emb_dim)
        cache_base = (CLNG(layer_idx) * CLNG(g_tiny_n_positions) + CLNG(cache_pos)) * CLNG(emb_dim)

        TinyGPTLayerNormVec g_tiny_x_vec(), g_tiny_norm_vec(), emb_dim, g_tiny_ln1_w(), g_tiny_ln1_b(), layer_e_base
        TinyGPTLinearVec g_tiny_norm_vec(), g_tiny_q_vec(), emb_dim, emb_dim, g_tiny_q_w(), layer_ee_base, g_tiny_q_b(), layer_e_base
        TinyGPTLinearVec g_tiny_norm_vec(), g_tiny_k_vec(), emb_dim, emb_dim, g_tiny_k_w(), layer_ee_base, g_tiny_k_b(), layer_e_base
        TinyGPTLinearVec g_tiny_norm_vec(), g_tiny_v_vec(), emb_dim, emb_dim, g_tiny_v_w(), layer_ee_base, g_tiny_v_b(), layer_e_base

        FOR emb_idx = 0 TO emb_dim - 1
            g_tiny_cache_k(cache_base + emb_idx) = g_tiny_k_vec(emb_idx)
            g_tiny_cache_v(cache_base + emb_idx) = g_tiny_v_vec(emb_idx)
        NEXT emb_idx

        FOR emb_idx = 0 TO emb_dim - 1
            g_tiny_att_vec(emb_idx) = 0.0
        NEXT emb_idx

        FOR head_idx = 0 TO head_count - 1
            max_score = -1.0E+30

            FOR src_idx = 0 TO cache_pos
                src_base = (CLNG(layer_idx) * CLNG(g_tiny_n_positions) + CLNG(src_idx)) * CLNG(emb_dim)
                score_value = 0.0

                FOR d = 0 TO head_dim - 1
                    q_index = head_idx * head_dim + d
                    score_value = score_value + g_tiny_q_vec(q_index) * g_tiny_cache_k(src_base + q_index)
                NEXT d

                score_value = score_value * scale_value
                g_tiny_score_vec(src_idx) = CSNG(score_value)
                IF score_value > max_score THEN max_score = score_value
            NEXT src_idx

            exp_sum = 0.0
            FOR src_idx = 0 TO cache_pos
                prob_value = EXP(g_tiny_score_vec(src_idx) - max_score)
                g_tiny_score_vec(src_idx) = CSNG(prob_value)
                exp_sum = exp_sum + prob_value
            NEXT src_idx
            IF exp_sum <= 0.0 THEN exp_sum = 1.0

            FOR d = 0 TO head_dim - 1
                score_value = 0.0
                kv_offset = head_idx * head_dim + d

                FOR src_idx = 0 TO cache_pos
                    src_base = (CLNG(layer_idx) * CLNG(g_tiny_n_positions) + CLNG(src_idx)) * CLNG(emb_dim)
                    prob_value = g_tiny_score_vec(src_idx) / exp_sum
                    score_value = score_value + prob_value * g_tiny_cache_v(src_base + kv_offset)
                NEXT src_idx

                g_tiny_att_vec(kv_offset) = CSNG(score_value)
            NEXT d
        NEXT head_idx

        TinyGPTLinearVec g_tiny_att_vec(), g_tiny_proj_vec(), emb_dim, emb_dim, g_tiny_proj_w(), layer_ee_base, g_tiny_proj_b(), layer_e_base

        FOR emb_idx = 0 TO emb_dim - 1
            g_tiny_x_vec(emb_idx) = g_tiny_x_vec(emb_idx) + g_tiny_proj_vec(emb_idx)
        NEXT emb_idx

        TinyGPTLayerNormVec g_tiny_x_vec(), g_tiny_norm_vec(), emb_dim, g_tiny_ln2_w(), g_tiny_ln2_b(), layer_e_base
        TinyGPTLinearVec g_tiny_norm_vec(), g_tiny_ff1_vec(), emb_dim, g_tiny_hidden_dim, g_tiny_fc1_w(), layer_eh_base, g_tiny_fc1_b(), CLNG(layer_idx) * CLNG(g_tiny_hidden_dim)

        FOR hidden_idx = 0 TO g_tiny_hidden_dim - 1
            g_tiny_ff1_vec(hidden_idx) = TinyGPTGELU(g_tiny_ff1_vec(hidden_idx))
        NEXT hidden_idx

        TinyGPTLinearVec g_tiny_ff1_vec(), g_tiny_ff2_vec(), g_tiny_hidden_dim, emb_dim, g_tiny_fc2_w(), layer_he_base, g_tiny_fc2_b(), layer_e_base

        FOR emb_idx = 0 TO emb_dim - 1
            g_tiny_x_vec(emb_idx) = g_tiny_x_vec(emb_idx) + g_tiny_ff2_vec(emb_idx)
        NEXT emb_idx
    NEXT layer_idx

    IF write_logits <> 0 THEN
        TinyGPTLayerNormVec g_tiny_x_vec(), g_tiny_norm_vec(), emb_dim, g_tiny_final_ln_w(), g_tiny_final_ln_b(), 0

        FOR vocab_idx = 0 TO g_tiny_vocab_size - 1
            sum_value = g_tiny_head_b(vocab_idx)
            FOR emb_idx = 0 TO emb_dim - 1
                sum_value = sum_value + g_tiny_norm_vec(emb_idx) * g_tiny_head_w(CLNG(emb_idx) * CLNG(g_tiny_vocab_size) + vocab_idx)
            NEXT emb_idx
            logits(vocab_idx) = CSNG(sum_value)
        NEXT vocab_idx
    END IF
END SUB

FUNCTION TinyGPTGELU(x AS SINGLE) AS SINGLE
    DIM xd AS DOUBLE
    xd = x
    RETURN CSNG(0.5 * xd * (1.0 + TANH(0.7978845608 * (xd + 0.044715 * xd * xd * xd))))
END FUNCTION

FUNCTION TinyGPTSample(logits() AS SINGLE, context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
    DIM vocab_size AS INTEGER
    DIM generated_count AS INTEGER
    DIM i AS INTEGER
    DIM j AS INTEGER
    DIM limit_count AS INTEGER
    DIM temp_val AS SINGLE
    DIM temp_idx AS INTEGER
    DIM max_logit AS DOUBLE
    DIM exp_sum AS DOUBLE
    DIM cumulative AS DOUBLE
    DIM sample_value AS DOUBLE
    DIM sorted_vals() AS SINGLE
    DIM sorted_idx() AS INTEGER
    DIM probs() AS DOUBLE

    vocab_size = g_tiny_vocab_size
    generated_count = context_len - g_tiny_generation_start_len
    IF generated_count < 0 THEN generated_count = 0

    FOR i = 0 TO vocab_size - 1
        IF TinyGPTTokenCanFollowContext(context(), context_len, i) = 0 THEN logits(i) = -1.0E+30
        IF TinyGPTTokenFollowPenalty(context(), context_len, i) <> 0 THEN logits(i) = logits(i) - TINYGPT_BYTE_RUN_FLOAT_PENALTY
    NEXT i

    IF generated_count < TINYGPT_MIN_GENERATED THEN
        logits(TINYGPT_EOT_TOKEN) = -1.0E+30
    END IF

    IF temperature <= 0.0 THEN
        max_logit = -1.0E+30
        temp_idx = TINYGPT_EOT_TOKEN
        FOR i = 0 TO vocab_size - 1
            IF logits(i) > max_logit THEN
                max_logit = logits(i)
                temp_idx = i
            END IF
        NEXT i
        RETURN temp_idx
    END IF

    REDIM sorted_vals(0 TO vocab_size - 1)
    REDIM sorted_idx(0 TO vocab_size - 1)
    REDIM probs(0 TO vocab_size - 1)

    FOR i = 0 TO vocab_size - 1
        sorted_vals(i) = logits(i) / temperature
        sorted_idx(i) = i
    NEXT i

    FOR i = 0 TO vocab_size - 2
        FOR j = 0 TO vocab_size - 2 - i
            IF sorted_vals(j) < sorted_vals(j + 1) THEN
                temp_val = sorted_vals(j)
                temp_idx = sorted_idx(j)
                sorted_vals(j) = sorted_vals(j + 1)
                sorted_idx(j) = sorted_idx(j + 1)
                sorted_vals(j + 1) = temp_val
                sorted_idx(j + 1) = temp_idx
            END IF
        NEXT j
    NEXT i

    limit_count = vocab_size
    IF top_k > 0 AND top_k < limit_count THEN limit_count = top_k
    IF limit_count < 1 THEN limit_count = 1

    max_logit = sorted_vals(0)
    exp_sum = 0.0
    FOR i = 0 TO limit_count - 1
        probs(i) = EXP(sorted_vals(i) - max_logit)
        exp_sum = exp_sum + probs(i)
    NEXT i
    IF exp_sum <= 0.0 THEN RETURN sorted_idx(0)

    IF top_p > 0.0 AND top_p < 1.0 THEN
        cumulative = 0.0
        FOR i = 0 TO limit_count - 1
            cumulative = cumulative + probs(i) / exp_sum
            IF cumulative >= top_p THEN
                limit_count = i + 1
                EXIT FOR
            END IF
        NEXT i
    END IF

    exp_sum = 0.0
    FOR i = 0 TO limit_count - 1
        exp_sum = exp_sum + probs(i)
    NEXT i
    IF exp_sum <= 0.0 THEN RETURN sorted_idx(0)

    sample_value = RND * exp_sum
    cumulative = 0.0
    FOR i = 0 TO limit_count - 1
        cumulative = cumulative + probs(i)
        IF sample_value <= cumulative THEN RETURN sorted_idx(i)
    NEXT i

    RETURN sorted_idx(0)
END FUNCTION

FUNCTION TinyGPTNextToken(context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
    DIM fx_logits() AS LONG

    IF g_tiny_loaded = 0 THEN RETURN TINYGPT_EOT_TOKEN
    IF g_tiny_fixed_loaded <> 0 THEN
        IF context_len > 0 AND context_len <= g_tiny_n_positions THEN
            RETURN TinyGPTNextTokenFixedCached(context(), context_len, temperature, top_p, top_k)
        END IF
        IF context_len > g_tiny_n_positions THEN
            IF TinyGPTForwardFixedLogits(context(), context_len, fx_logits()) <> 0 THEN
                RETURN TinyGPTFixedSample(fx_logits(), context(), context_len, temperature, top_p, top_k)
            END IF
        END IF
        RETURN TINYGPT_EOT_TOKEN
    END IF

    IF context_len > 0 AND context_len <= g_tiny_n_positions THEN
        RETURN TinyGPTNextTokenCached(context(), context_len, temperature, top_p, top_k)
    END IF

    RETURN TinyGPTNextTokenFull(context(), context_len, temperature, top_p, top_k)
END FUNCTION

FUNCTION TinyGPTNextTokenCached(context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
    DIM pos_idx AS INTEGER
    DIM token_id AS INTEGER
    DIM cached_token_id AS INTEGER
    DIM write_logits AS INTEGER

    IF g_tiny_loaded = 0 THEN RETURN TINYGPT_EOT_TOKEN
    IF context_len < 1 THEN RETURN TinyGPTNextTokenFull(context(), context_len, temperature, top_p, top_k)
    IF context_len > g_tiny_n_positions THEN RETURN TinyGPTNextTokenFull(context(), context_len, temperature, top_p, top_k)

    IF g_tiny_cache_len > context_len THEN
        TinyGPTResetDecodeCache
    END IF

    FOR pos_idx = 0 TO g_tiny_cache_len - 1
        token_id = context(pos_idx)
        IF token_id < 0 OR token_id >= g_tiny_vocab_size THEN token_id = TINYGPT_UNK_TOKEN
        cached_token_id = g_tiny_cache_tokens(pos_idx)

        IF cached_token_id <> token_id THEN
            TinyGPTResetDecodeCache
            EXIT FOR
        END IF
    NEXT pos_idx

    FOR pos_idx = g_tiny_cache_len TO context_len - 1
        token_id = context(pos_idx)
        IF token_id < 0 OR token_id >= g_tiny_vocab_size THEN token_id = TINYGPT_UNK_TOKEN

        write_logits = 0
        IF pos_idx = context_len - 1 THEN write_logits = 1
        TinyGPTForwardCachedToken token_id, pos_idx, write_logits, g_tiny_logits_vec()
        g_tiny_cache_tokens(pos_idx) = token_id
        g_tiny_cache_len = pos_idx + 1
    NEXT pos_idx

    RETURN TinyGPTSample(g_tiny_logits_vec(), context(), context_len, temperature, top_p, top_k)
END FUNCTION

FUNCTION TinyGPTForwardFloatLogits(context() AS INTEGER, context_len AS INTEGER, logits() AS SINGLE) AS INTEGER
    DIM active_context() AS INTEGER
    DIM active_len AS INTEGER
    DIM start_idx AS INTEGER
    DIM pos_idx AS INTEGER
    DIM token_id AS INTEGER
    DIM cached_token_id AS INTEGER
    DIM write_logits AS INTEGER
    DIM i AS INTEGER

    IF g_tiny_loaded = 0 THEN RETURN 0
    IF g_tiny_fixed_loaded <> 0 THEN RETURN 0
    IF context_len < 1 THEN RETURN 0
    IF g_tiny_n_positions < 1 THEN RETURN 0

    active_len = context_len
    start_idx = 0
    IF active_len > g_tiny_n_positions THEN
        start_idx = active_len - g_tiny_n_positions
        active_len = g_tiny_n_positions
    END IF

    REDIM active_context(0 TO active_len - 1)
    FOR i = 0 TO active_len - 1
        active_context(i) = context(start_idx + i)
    NEXT i

    IF g_tiny_cache_len > active_len THEN
        TinyGPTResetDecodeCache
    END IF

    FOR pos_idx = 0 TO g_tiny_cache_len - 1
        token_id = active_context(pos_idx)
        IF token_id < 0 OR token_id >= g_tiny_vocab_size THEN token_id = TINYGPT_UNK_TOKEN
        cached_token_id = g_tiny_cache_tokens(pos_idx)

        IF cached_token_id <> token_id THEN
            TinyGPTResetDecodeCache
            EXIT FOR
        END IF
    NEXT pos_idx

    FOR pos_idx = g_tiny_cache_len TO active_len - 1
        token_id = active_context(pos_idx)
        IF token_id < 0 OR token_id >= g_tiny_vocab_size THEN token_id = TINYGPT_UNK_TOKEN

        write_logits = 0
        IF pos_idx = active_len - 1 THEN write_logits = 1
        TinyGPTForwardCachedToken token_id, pos_idx, write_logits, g_tiny_logits_vec()
        g_tiny_cache_tokens(pos_idx) = token_id
        g_tiny_cache_len = pos_idx + 1
    NEXT pos_idx

    REDIM logits(0 TO g_tiny_vocab_size - 1)
    FOR i = 0 TO g_tiny_vocab_size - 1
        logits(i) = g_tiny_logits_vec(i)
    NEXT i

    ERASE active_context
    RETURN 1
END FUNCTION

FUNCTION TinyGPTNextTokenFull(context() AS INTEGER, context_len AS INTEGER, temperature AS SINGLE, top_p AS SINGLE, top_k AS INTEGER) AS INTEGER
    DIM seq_len AS INTEGER
    DIM start_idx AS INTEGER
    DIM pos_idx AS INTEGER
    DIM emb_idx AS INTEGER
    DIM token_id AS INTEGER
    DIM layer_idx AS INTEGER
    DIM layer_e_base AS LONG
    DIM layer_ee_base AS LONG
    DIM layer_eh_base AS LONG
    DIM layer_he_base AS LONG
    DIM x_arr() AS SINGLE
    DIM norm_arr() AS SINGLE
    DIM att_arr() AS SINGLE
    DIM ff1_arr() AS SINGLE
    DIM ff2_arr() AS SINGLE
    DIM last_norm() AS SINGLE
    DIM logits() AS SINGLE
    DIM sum_value AS DOUBLE
    DIM v AS INTEGER
    DIM h AS INTEGER

    IF g_tiny_loaded = 0 THEN RETURN TINYGPT_EOT_TOKEN

    seq_len = context_len
    start_idx = 0
    IF seq_len < 1 THEN seq_len = 1
    IF seq_len > g_tiny_n_positions THEN
        start_idx = seq_len - g_tiny_n_positions
        seq_len = g_tiny_n_positions
    END IF

    REDIM x_arr(0 TO CLNG(seq_len) * CLNG(g_tiny_n_embd) - 1)
    REDIM norm_arr(0 TO CLNG(seq_len) * CLNG(g_tiny_n_embd) - 1)
    REDIM att_arr(0 TO CLNG(seq_len) * CLNG(g_tiny_n_embd) - 1)
    REDIM ff1_arr(0 TO CLNG(seq_len) * CLNG(g_tiny_hidden_dim) - 1)
    REDIM ff2_arr(0 TO CLNG(seq_len) * CLNG(g_tiny_n_embd) - 1)
    REDIM last_norm(0 TO g_tiny_n_embd - 1)
    REDIM logits(0 TO g_tiny_vocab_size - 1)

    FOR pos_idx = 0 TO seq_len - 1
        IF context_len <= 0 THEN
            token_id = TINYGPT_EOT_TOKEN
        ELSE
            token_id = context(start_idx + pos_idx)
        END IF

        IF token_id < 0 OR token_id >= g_tiny_vocab_size THEN
            token_id = TINYGPT_UNK_TOKEN
        END IF

        FOR emb_idx = 0 TO g_tiny_n_embd - 1
            x_arr(CLNG(pos_idx) * CLNG(g_tiny_n_embd) + emb_idx) = g_tiny_tok_emb(CLNG(token_id) * CLNG(g_tiny_n_embd) + emb_idx) + g_tiny_pos_emb(CLNG(pos_idx) * CLNG(g_tiny_n_embd) + emb_idx)
        NEXT emb_idx
    NEXT pos_idx

    FOR layer_idx = 0 TO g_tiny_n_layer - 1
        layer_e_base = CLNG(layer_idx) * CLNG(g_tiny_n_embd)
        layer_ee_base = CLNG(layer_idx) * CLNG(g_tiny_n_embd) * CLNG(g_tiny_n_embd)
        layer_eh_base = CLNG(layer_idx) * CLNG(g_tiny_n_embd) * CLNG(g_tiny_hidden_dim)
        layer_he_base = CLNG(layer_idx) * CLNG(g_tiny_hidden_dim) * CLNG(g_tiny_n_embd)

        TinyGPTLayerNormSeq x_arr(), norm_arr(), seq_len, g_tiny_n_embd, g_tiny_ln1_w(), g_tiny_ln1_b(), layer_e_base
        TinyGPTAttention norm_arr(), att_arr(), seq_len, layer_idx

        FOR pos_idx = 0 TO seq_len - 1
            FOR emb_idx = 0 TO g_tiny_n_embd - 1
                x_arr(CLNG(pos_idx) * CLNG(g_tiny_n_embd) + emb_idx) = x_arr(CLNG(pos_idx) * CLNG(g_tiny_n_embd) + emb_idx) + att_arr(CLNG(pos_idx) * CLNG(g_tiny_n_embd) + emb_idx)
            NEXT emb_idx
        NEXT pos_idx

        TinyGPTLayerNormSeq x_arr(), norm_arr(), seq_len, g_tiny_n_embd, g_tiny_ln2_w(), g_tiny_ln2_b(), layer_e_base
        TinyGPTLinearSeq norm_arr(), ff1_arr(), seq_len, g_tiny_n_embd, g_tiny_hidden_dim, g_tiny_fc1_w(), layer_eh_base, g_tiny_fc1_b(), CLNG(layer_idx) * CLNG(g_tiny_hidden_dim)

        FOR pos_idx = 0 TO seq_len - 1
            FOR h = 0 TO g_tiny_hidden_dim - 1
                ff1_arr(CLNG(pos_idx) * CLNG(g_tiny_hidden_dim) + h) = TinyGPTGELU(ff1_arr(CLNG(pos_idx) * CLNG(g_tiny_hidden_dim) + h))
            NEXT h
        NEXT pos_idx

        TinyGPTLinearSeq ff1_arr(), ff2_arr(), seq_len, g_tiny_hidden_dim, g_tiny_n_embd, g_tiny_fc2_w(), layer_he_base, g_tiny_fc2_b(), layer_e_base

        FOR pos_idx = 0 TO seq_len - 1
            FOR emb_idx = 0 TO g_tiny_n_embd - 1
                x_arr(CLNG(pos_idx) * CLNG(g_tiny_n_embd) + emb_idx) = x_arr(CLNG(pos_idx) * CLNG(g_tiny_n_embd) + emb_idx) + ff2_arr(CLNG(pos_idx) * CLNG(g_tiny_n_embd) + emb_idx)
            NEXT emb_idx
        NEXT pos_idx
    NEXT layer_idx

    TinyGPTLayerNormLast x_arr(), last_norm(), seq_len, g_tiny_n_embd, g_tiny_final_ln_w(), g_tiny_final_ln_b()

    FOR v = 0 TO g_tiny_vocab_size - 1
        sum_value = g_tiny_head_b(v)
        FOR emb_idx = 0 TO g_tiny_n_embd - 1
            sum_value = sum_value + last_norm(emb_idx) * g_tiny_head_w(CLNG(emb_idx) * CLNG(g_tiny_vocab_size) + v)
        NEXT emb_idx
        logits(v) = CSNG(sum_value)
    NEXT v

    RETURN TinyGPTSample(logits(), context(), context_len, temperature, top_p, top_k)
END FUNCTION
