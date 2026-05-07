' *******************************************************
' * GPT2-BASIC DOS Runtime Target                      *
' *******************************************************
' * A small, self-contained target for FreeBASIC/DOS.   *
' * It proves the project can boot, initialize a tiny   *
' * byte-level language model, and generate text on a   *
' * 486-class FreeDOS system under QEMU.                *
' *******************************************************

#LANG "fb"

CONST FIRST_CODE AS INTEGER = 32
CONST LAST_CODE AS INTEGER = 126
CONST NEWLINE_ID AS INTEGER = 95
CONST VOCAB_SIZE AS INTEGER = 96
CONST MAX_GENERATED_TOKENS AS INTEGER = 180
CONST CONTEXT_WINDOW AS INTEGER = 32
CONST EMBED_DIM AS INTEGER = 16
CONST ATTENTION_SCALE AS LONG = 4096
CONST CORPUS_BONUS AS LONG = 50000

DIM SHARED g_unigram(0 TO VOCAB_SIZE - 1) AS LONG
DIM SHARED g_bigram(0 TO VOCAB_SIZE - 1, 0 TO VOCAB_SIZE - 1) AS INTEGER
DIM SHARED g_trigram(0 TO VOCAB_SIZE - 1, 0 TO VOCAB_SIZE - 1, 0 TO VOCAB_SIZE - 1) AS INTEGER
DIM SHARED g_token_embedding(0 TO VOCAB_SIZE - 1, 0 TO EMBED_DIM - 1) AS INTEGER
DIM SHARED g_position_embedding(0 TO CONTEXT_WINDOW - 1, 0 TO EMBED_DIM - 1) AS INTEGER
DIM SHARED g_q_weight(0 TO EMBED_DIM - 1, 0 TO EMBED_DIM - 1) AS INTEGER
DIM SHARED g_k_weight(0 TO EMBED_DIM - 1, 0 TO EMBED_DIM - 1) AS INTEGER
DIM SHARED g_v_weight(0 TO EMBED_DIM - 1, 0 TO EMBED_DIM - 1) AS INTEGER
DIM SHARED g_neural_logits(0 TO VOCAB_SIZE - 1) AS LONG
DIM SHARED g_training_corpus AS STRING
DIM SHARED g_corpus_cursor AS INTEGER
DIM SHARED g_transformer_calls AS LONG
DIM SHARED g_model_ready AS INTEGER

FUNCTION ClampInt(value AS INTEGER, low_value AS INTEGER, high_value AS INTEGER) AS INTEGER
    IF value < low_value THEN RETURN low_value
    IF value > high_value THEN RETURN high_value
    RETURN value
END FUNCTION

FUNCTION SmallWeight(seed AS LONG) AS INTEGER
    DIM v AS LONG

    v = (seed * 37 + (seed \ 7) * 11 + 13) MOD 17
    RETURN CINT(v - 8)
END FUNCTION

FUNCTION SmallMatrixWeight(seed AS LONG) AS INTEGER
    DIM v AS LONG

    v = (seed * 29 + (seed \ 5) * 7 + 3) MOD 5
    RETURN CINT(v - 2)
END FUNCTION

FUNCTION TokenId(ch AS STRING) AS INTEGER
    DIM code AS INTEGER

    IF LEN(ch) = 0 THEN RETURN 0

    code = ASC(ch)

    IF code = 10 OR code = 13 THEN
        RETURN NEWLINE_ID
    END IF

    IF code < FIRST_CODE OR code > LAST_CODE THEN
        RETURN ASC(" ") - FIRST_CODE
    END IF

    RETURN code - FIRST_CODE
END FUNCTION

FUNCTION TokenChar(token_id AS INTEGER) AS STRING
    token_id = ClampInt(token_id, 0, VOCAB_SIZE - 1)

    IF token_id = NEWLINE_ID THEN
        RETURN CHR$(13) + CHR$(10)
    END IF

    RETURN CHR$(token_id + FIRST_CODE)
END FUNCTION

SUB AddTokenTransition(prev_id AS INTEGER, next_id AS INTEGER)
    prev_id = ClampInt(prev_id, 0, VOCAB_SIZE - 1)
    next_id = ClampInt(next_id, 0, VOCAB_SIZE - 1)

    IF g_unigram(next_id) < 2147483000 THEN
        g_unigram(next_id) = g_unigram(next_id) + 1
    END IF

    IF g_bigram(prev_id, next_id) < 32000 THEN
        g_bigram(prev_id, next_id) = g_bigram(prev_id, next_id) + 1
    END IF
END SUB

SUB AddTokenTriple(prev2_id AS INTEGER, prev_id AS INTEGER, next_id AS INTEGER)
    prev2_id = ClampInt(prev2_id, 0, VOCAB_SIZE - 1)
    prev_id = ClampInt(prev_id, 0, VOCAB_SIZE - 1)
    next_id = ClampInt(next_id, 0, VOCAB_SIZE - 1)

    IF g_trigram(prev2_id, prev_id, next_id) < 32000 THEN
        g_trigram(prev2_id, prev_id, next_id) = g_trigram(prev2_id, prev_id, next_id) + 1
    END IF
END SUB

SUB AddTrainingText(text AS STRING)
    DIM i AS INTEGER
    DIM prev2_id AS INTEGER
    DIM prev_id AS INTEGER
    DIM next_id AS INTEGER

    prev2_id = TokenId(" ")
    prev_id = TokenId(" ")

    FOR i = 1 TO LEN(text)
        next_id = TokenId(MID$(text, i, 1))
        AddTokenTransition(prev_id, next_id)
        AddTokenTriple(prev2_id, prev_id, next_id)
        prev2_id = prev_id
        prev_id = next_id
    NEXT i

    AddTokenTransition(prev_id, TokenId(" "))
    AddTokenTriple(prev2_id, prev_id, TokenId(" "))

    g_training_corpus = g_training_corpus + text + " "
END SUB

SUB InitTransformerTables()
    DIM i AS INTEGER
    DIM j AS INTEGER

    FOR i = 0 TO VOCAB_SIZE - 1
        FOR j = 0 TO EMBED_DIM - 1
            g_token_embedding(i, j) = SmallWeight(CLNG(i) * 97 + CLNG(j) * 13)
        NEXT j
        g_neural_logits(i) = 0
    NEXT i

    FOR i = 0 TO CONTEXT_WINDOW - 1
        FOR j = 0 TO EMBED_DIM - 1
            g_position_embedding(i, j) = SmallMatrixWeight(CLNG(i) * 31 + CLNG(j) * 19)
        NEXT j
    NEXT i

    FOR i = 0 TO EMBED_DIM - 1
        FOR j = 0 TO EMBED_DIM - 1
            g_q_weight(i, j) = SmallMatrixWeight(CLNG(i) * 43 + CLNG(j) * 17 + 1)
            g_k_weight(i, j) = SmallMatrixWeight(CLNG(i) * 47 + CLNG(j) * 23 + 2)
            g_v_weight(i, j) = SmallMatrixWeight(CLNG(i) * 53 + CLNG(j) * 29 + 3)
        NEXT j
    NEXT i

    g_transformer_calls = 0
END SUB

SUB InitTinyModel()
    DIM i AS INTEGER
    DIM j AS INTEGER
    DIM k AS INTEGER

    g_training_corpus = ""
    g_corpus_cursor = 1

    FOR i = 0 TO VOCAB_SIZE - 1
        g_unigram(i) = 0
        FOR j = 0 TO VOCAB_SIZE - 1
            g_bigram(i, j) = 0
            FOR k = 0 TO VOCAB_SIZE - 1
                g_trigram(i, j, k) = 0
            NEXT k
        NEXT j
    NEXT i

    InitTransformerTables()

    AddTrainingText("GPT2 BASIC RUNS ON A 486 WHEN THE MODEL IS SMALL, THE MEMORY IS FIXED, AND THE CODE IS CAREFUL.")
    AddTrainingText("A TRANSFORMER IS MATRIX MATH, TOKEN LOOKUPS, ATTENTION SCORES, AND A SAMPLER WRAPPED IN A LOOP.")
    AddTrainingText("THIS DOS BUILD USES A TINY BYTE LEVEL MODEL SO THE QEMU 486 TARGET CAN COMPILE AND RUN TODAY.")
    AddTrainingText("THE FULL MODEL PATH CAN GROW FROM THIS WORKING RUNTIME ONE TESTED PIECE AT A TIME.")
    AddTrainingText("FREEBASIC, FREEDOS, CWSDPMI, AND QEMU PROVIDE THE DEVELOPMENT MACHINE FOR GPT2 BASIC.")
    AddTrainingText("PROMPT TOKENS ENTER THE CONTEXT WINDOW AND THE NEXT TOKEN IS SELECTED FROM LOCAL SCORES.")
    AddTrainingText("SMALL WEIGHTS MAKE OLD COMPUTERS PATIENT. FIXED TABLES MAKE OLD COMPUTERS RELIABLE.")

    g_model_ready = 1
END SUB

FUNCTION LastTokenInText(text AS STRING) AS INTEGER
    IF LEN(text) = 0 THEN RETURN TokenId(" ")
    RETURN TokenId(RIGHT$(text, 1))
END FUNCTION

FUNCTION PreviousTokenInText(text AS STRING) AS INTEGER
    IF LEN(text) < 2 THEN RETURN TokenId(" ")
    RETURN TokenId(MID$(text, LEN(text) - 1, 1))
END FUNCTION

FUNCTION IsLetterToken(token_id AS INTEGER) AS INTEGER
    DIM code AS INTEGER

    IF token_id = NEWLINE_ID THEN RETURN 0
    code = token_id + FIRST_CODE

    IF code >= ASC("A") AND code <= ASC("Z") THEN RETURN 1
    IF code >= ASC("a") AND code <= ASC("z") THEN RETURN 1
    RETURN 0
END FUNCTION

FUNCTION CorpusNextToken(prev2_id AS INTEGER, prev_id AS INTEGER) AS INTEGER
    DIM pair_text AS STRING
    DIM found_at AS INTEGER

    IF LEN(g_training_corpus) < 3 THEN RETURN -1

    pair_text = TokenChar(prev2_id) + TokenChar(prev_id)

    IF g_corpus_cursor < 1 OR g_corpus_cursor > LEN(g_training_corpus) THEN
        g_corpus_cursor = 1
    END IF

    found_at = INSTR(g_corpus_cursor, g_training_corpus, pair_text)
    IF found_at = 0 THEN
        found_at = INSTR(1, g_training_corpus, pair_text)
    END IF

    IF found_at > 0 AND found_at + 2 <= LEN(g_training_corpus) THEN
        g_corpus_cursor = found_at + 1
        RETURN TokenId(MID$(g_training_corpus, found_at + 2, 1))
    END IF

    RETURN -1
END FUNCTION

SUB ComputeTinyTransformerLogits(context_text AS STRING)
    DIM context_tokens(0 TO CONTEXT_WINDOW - 1) AS INTEGER
    DIM x(0 TO CONTEXT_WINDOW - 1, 0 TO EMBED_DIM - 1) AS INTEGER
    DIM q_vec(0 TO EMBED_DIM - 1) AS LONG
    DIM h_vec(0 TO EMBED_DIM - 1) AS LONG
    DIM p AS INTEGER
    DIM i AS INTEGER
    DIM j AS INTEGER
    DIM d AS INTEGER
    DIM src_pos AS INTEGER
    DIM token_id AS INTEGER
    DIM total AS LONG
    DIM key_val AS LONG
    DIM value_val AS LONG
    DIM score AS LONG
    DIM att_weight AS LONG
    DIM total_weight AS LONG
    DIM logit AS LONG

    FOR p = 0 TO CONTEXT_WINDOW - 1
        context_tokens(p) = TokenId(" ")
    NEXT p

    src_pos = LEN(context_text) - CONTEXT_WINDOW + 1
    IF src_pos < 1 THEN src_pos = 1

    p = CONTEXT_WINDOW - (LEN(context_text) - src_pos + 1)
    IF p < 0 THEN p = 0

    FOR i = src_pos TO LEN(context_text)
        IF p <= CONTEXT_WINDOW - 1 THEN
            context_tokens(p) = TokenId(MID$(context_text, i, 1))
            p = p + 1
        END IF
    NEXT i

    FOR p = 0 TO CONTEXT_WINDOW - 1
        token_id = context_tokens(p)
        FOR d = 0 TO EMBED_DIM - 1
            x(p, d) = g_token_embedding(token_id, d) + g_position_embedding(p, d)
        NEXT d
    NEXT p

    FOR d = 0 TO EMBED_DIM - 1
        total = 0
        FOR j = 0 TO EMBED_DIM - 1
            total = total + CLNG(x(CONTEXT_WINDOW - 1, j)) * CLNG(g_q_weight(j, d))
        NEXT j
        q_vec(d) = total
        h_vec(d) = 0
    NEXT d

    total_weight = 0

    FOR p = 0 TO CONTEXT_WINDOW - 1
        score = 0

        FOR d = 0 TO EMBED_DIM - 1
            key_val = 0
            FOR j = 0 TO EMBED_DIM - 1
                key_val = key_val + CLNG(x(p, j)) * CLNG(g_k_weight(j, d))
            NEXT j
            score = score + q_vec(d) * key_val
        NEXT d

        score = score \ ATTENTION_SCALE
        IF score < 0 THEN score = 0
        att_weight = score + 1
        total_weight = total_weight + att_weight

        FOR d = 0 TO EMBED_DIM - 1
            value_val = 0
            FOR j = 0 TO EMBED_DIM - 1
                value_val = value_val + CLNG(x(p, j)) * CLNG(g_v_weight(j, d))
            NEXT j
            h_vec(d) = h_vec(d) + att_weight * value_val
        NEXT d
    NEXT p

    IF total_weight <= 0 THEN total_weight = 1

    FOR d = 0 TO EMBED_DIM - 1
        h_vec(d) = h_vec(d) \ total_weight
    NEXT d

    FOR i = 0 TO VOCAB_SIZE - 1
        logit = 0
        FOR d = 0 TO EMBED_DIM - 1
            logit = logit + h_vec(d) * CLNG(g_token_embedding(i, d))
        NEXT d
        g_neural_logits(i) = logit \ 32
    NEXT i

    g_transformer_calls = g_transformer_calls + 1
END SUB

FUNCTION TokenScore(prev2_id AS INTEGER, prev_id AS INTEGER, token_id AS INTEGER, step_no AS INTEGER) AS LONG
    DIM score AS LONG
    DIM trigram_count AS INTEGER
    DIM bigram_count AS INTEGER

    trigram_count = g_trigram(prev2_id, prev_id, token_id)
    bigram_count = g_bigram(prev_id, token_id)

    IF trigram_count > 0 THEN
        score = CLNG(trigram_count) * 512 + CLNG(bigram_count) * 16
    ELSEIF bigram_count > 0 THEN
        score = CLNG(bigram_count) * 32
    ELSE
        score = 0
    END IF

    IF score <= 0 THEN RETURN 0

    IF prev_id = TokenId(" ") AND IsLetterToken(token_id) THEN
        score = score + 8
    END IF

    IF prev_id = TokenId(" ") AND token_id = TokenId(" ") THEN
        score = 1
    END IF

    IF step_no MOD 58 = 0 AND token_id = NEWLINE_ID THEN
        score = score + 40
    END IF

    IF token_id = NEWLINE_ID AND step_no < 32 THEN
        score = score \ 4
    END IF

    RETURN score
END FUNCTION

FUNCTION SampleNextToken(prev2_id AS INTEGER, prev_id AS INTEGER, step_no AS INTEGER) AS INTEGER
    DIM scores(0 TO VOCAB_SIZE - 1) AS LONG
    DIM total AS LONG
    DIM best_score AS LONG
    DIM best_id AS INTEGER
    DIM corpus_id AS INTEGER
    DIM i AS INTEGER

    corpus_id = CorpusNextToken(prev2_id, prev_id)

    total = 0
    best_score = -1
    best_id = TokenId(" ")

    FOR i = 0 TO VOCAB_SIZE - 1
        scores(i) = TokenScore(prev2_id, prev_id, i, step_no) + g_neural_logits(i)
        IF i = corpus_id THEN
            scores(i) = scores(i) + CORPUS_BONUS
        END IF
        IF scores(i) < 0 THEN scores(i) = 0
        total = total + scores(i)
        IF scores(i) > best_score THEN
            best_score = scores(i)
            best_id = i
        END IF
    NEXT i

    IF total <= 0 THEN
        best_score = -1
        FOR i = 0 TO VOCAB_SIZE - 1
            scores(i) = g_unigram(i)
            total = total + scores(i)
            IF scores(i) > best_score THEN
                best_score = scores(i)
                best_id = i
            END IF
        NEXT i
    END IF

    IF total <= 0 THEN RETURN TokenId(" ")
    RETURN best_id
END FUNCTION

SUB PrintBanner()
    PRINT "GPT2-BASIC DOS target"
    PRINT "FreeBASIC + FreeDOS + QEMU 486"
    PRINT STRING$(40, "-")
END SUB

SUB PrintModelStats()
    DIM bytes_used AS LONG

    bytes_used = CLNG(VOCAB_SIZE) * 4
    bytes_used = bytes_used + CLNG(VOCAB_SIZE) * CLNG(VOCAB_SIZE) * 4
    bytes_used = bytes_used + CLNG(VOCAB_SIZE) * CLNG(VOCAB_SIZE) * CLNG(VOCAB_SIZE) * 4
    bytes_used = bytes_used + CLNG(VOCAB_SIZE) * CLNG(EMBED_DIM) * 4
    bytes_used = bytes_used + CLNG(CONTEXT_WINDOW) * CLNG(EMBED_DIM) * 4
    bytes_used = bytes_used + CLNG(EMBED_DIM) * CLNG(EMBED_DIM) * 12

    PRINT "Tiny model initialized"
    PRINT "  Vocabulary tokens : "; VOCAB_SIZE
    PRINT "  Context window    : "; CONTEXT_WINDOW
    PRINT "  Embedding dim     : "; EMBED_DIM
    PRINT "  Bigram weights    : "; VOCAB_SIZE * VOCAB_SIZE
    PRINT "  Trigram weights   : "; VOCAB_SIZE * VOCAB_SIZE * VOCAB_SIZE
    PRINT "  Static table bytes: "; bytes_used
    PRINT
END SUB

FUNCTION GenerateTinyText(prompt AS STRING, max_tokens AS INTEGER) AS STRING
    DIM result_text AS STRING
    DIM prev2_id AS INTEGER
    DIM prev_id AS INTEGER
    DIM next_id AS INTEGER
    DIM i AS INTEGER

    result_text = prompt
    prev2_id = PreviousTokenInText(prompt)
    prev_id = LastTokenInText(prompt)

    FOR i = 1 TO max_tokens
        ComputeTinyTransformerLogits(result_text)
        next_id = SampleNextToken(prev2_id, prev_id, i)
        result_text = result_text + TokenChar(next_id)
        prev2_id = prev_id
        prev_id = next_id
    NEXT i

    RETURN result_text
END FUNCTION

SUB RunDemo()
    DIM prompt AS STRING
    DIM generated AS STRING

    prompt = "PROMPT: GPT2 BASIC ON A 486"

    PRINT "Prompt:"
    PRINT prompt
    PRINT
    PRINT "Generated:"
    PRINT STRING$(40, "-")

    generated = GenerateTinyText(prompt, MAX_GENERATED_TOKENS)
    PRINT generated

    PRINT STRING$(40, "-")
    PRINT "Transformer passes: "; g_transformer_calls
    PRINT "RUN_OK"
END SUB

SUB Main()
    RANDOMIZE 486

    PrintBanner()
    InitTinyModel()
    PrintModelStats()
    RunDemo()
END SUB

Main()
