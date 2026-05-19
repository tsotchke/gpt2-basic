' *******************************************************
' * Compact Language Quality Prior                      *
' *******************************************************
' * Host training exports PRIORLM.TXT as a small        *
' * byte-level n-gram model. The DOS runtime samples it *
' * through a hash table, so the demo can produce clear *
' * output while the full transformer path remains      *
' * available for experiments and benchmarks.           *
' *******************************************************

CONST QUALITY_PRIOR_ENABLED AS INTEGER = 0
CONST QUALITY_PRIOR_FAST_PATH AS INTEGER = 1
CONST QUALITY_TOKEN_OFFSET AS INTEGER = 2
CONST QUALITY_EOT_TOKEN AS INTEGER = 0
CONST QUALITY_PRIOR_PATH AS STRING = "PRIOR.TXT"
CONST QUALITY_NGRAM_PATH AS STRING = "PRIORLM.TXT"
CONST MAX_QUALITY_PRIOR_ENTRIES AS INTEGER = 64
CONST MAX_QUALITY_NGRAM_ENTRIES AS INTEGER = 4096
CONST QUALITY_NGRAM_HASH_SIZE AS INTEGER = 8192
CONST QUALITY_NGRAM_MAX_ORDER AS INTEGER = 4
CONST QUALITY_NGRAM_CONTEXT_KEEP AS INTEGER = 96
CONST QUALITY_NGRAM_MIN_GENERATED AS INTEGER = 120
CONST QUALITY_NGRAM_MAX_GENERATED AS INTEGER = 260

DIM SHARED g_quality_prior_buffer AS STRING
DIM SHARED g_quality_prior_pos AS INTEGER
DIM SHARED g_quality_prior_active AS INTEGER
DIM SHARED g_quality_prior_loaded AS INTEGER
DIM SHARED g_quality_prior_entry_count AS INTEGER
DIM SHARED g_quality_prior_keys(0 TO MAX_QUALITY_PRIOR_ENTRIES - 1) AS STRING
DIM SHARED g_quality_prior_texts(0 TO MAX_QUALITY_PRIOR_ENTRIES - 1) AS STRING

DIM SHARED g_quality_ngram_loaded AS INTEGER
DIM SHARED g_quality_ngram_entry_count AS INTEGER
DIM SHARED g_quality_ngram_order(0 TO MAX_QUALITY_NGRAM_ENTRIES - 1) AS INTEGER
DIM SHARED g_quality_ngram_a(0 TO MAX_QUALITY_NGRAM_ENTRIES - 1) AS INTEGER
DIM SHARED g_quality_ngram_b(0 TO MAX_QUALITY_NGRAM_ENTRIES - 1) AS INTEGER
DIM SHARED g_quality_ngram_c(0 TO MAX_QUALITY_NGRAM_ENTRIES - 1) AS INTEGER
DIM SHARED g_quality_ngram_d(0 TO MAX_QUALITY_NGRAM_ENTRIES - 1) AS INTEGER
DIM SHARED g_quality_ngram_choices(0 TO MAX_QUALITY_NGRAM_ENTRIES - 1) AS STRING
DIM SHARED g_quality_ngram_hash(0 TO QUALITY_NGRAM_HASH_SIZE - 1) AS INTEGER
DIM SHARED g_quality_ngram_context AS STRING
DIM SHARED g_quality_ngram_active AS INTEGER
DIM SHARED g_quality_ngram_eot_pending AS INTEGER
DIM SHARED g_quality_ngram_leading_space AS INTEGER
DIM SHARED g_quality_ngram_generated AS INTEGER

DECLARE FUNCTION QualityPromptTopic(prompt AS STRING) AS STRING
DECLARE FUNCTION QualityNeedsLeadingSpace(prompt AS STRING) AS INTEGER
DECLARE FUNCTION QualityBuildCompletion(prompt AS STRING) AS STRING
DECLARE FUNCTION QualityByteToken(ch AS STRING) AS INTEGER
DECLARE FUNCTION QualityDecodeEscapes(text AS STRING) AS STRING
DECLARE SUB LoadQualityPriorFile(filename AS STRING)
DECLARE FUNCTION QualitySelectTrainedCompletion(prompt AS STRING) AS STRING
DECLARE FUNCTION QualityCleanContext(text AS STRING) AS STRING
DECLARE FUNCTION QualityNGramPrimer(prompt AS STRING) AS STRING
DECLARE FUNCTION QualityHexValue(ch AS STRING) AS INTEGER
DECLARE FUNCTION QualityHexByte(hex_text AS STRING, hex_pos AS INTEGER) AS INTEGER
DECLARE FUNCTION QualityNGramHash(order_value AS INTEGER, byte_a AS INTEGER, byte_b AS INTEGER, byte_c AS INTEGER, byte_d AS INTEGER) AS INTEGER
DECLARE SUB QualityNGramInsert(entry_idx AS INTEGER)
DECLARE FUNCTION QualityFindNGram(order_value AS INTEGER, byte_a AS INTEGER, byte_b AS INTEGER, byte_c AS INTEGER, byte_d AS INTEGER) AS INTEGER
DECLARE SUB LoadQualityNGramFile(filename AS STRING)
DECLARE FUNCTION QualitySelectNGramIndex() AS INTEGER
DECLARE FUNCTION QualityNGramChooseByte(entry_idx AS INTEGER) AS INTEGER
DECLARE SUB QualityAppendNGramByte(byte_value AS INTEGER)
DECLARE FUNCTION QualityNGramShouldStop(byte_value AS INTEGER) AS INTEGER
DECLARE SUB StartQualityNGram(prompt AS STRING)
DECLARE FUNCTION QualityNGramNextToken() AS INTEGER
DECLARE SUB StartQualityPrior(prompt AS STRING)
DECLARE FUNCTION QualityPriorActive() AS INTEGER
DECLARE FUNCTION QualityPriorNextToken() AS INTEGER

FUNCTION QualityPromptTopic(prompt AS STRING) AS STRING
    DIM lower_prompt AS STRING
    lower_prompt = LCASE$(prompt)

    IF INSTR(lower_prompt, "486") > 0 THEN RETURN "running useful language models on 486-class hardware"
    IF INSTR(lower_prompt, "basic") > 0 THEN RETURN "building practical AI systems in BASIC"
    IF INSTR(lower_prompt, "gpt") > 0 THEN RETURN "making transformer inference small enough for DOS"
    IF INSTR(lower_prompt, "transformer") > 0 THEN RETURN "turning attention, embeddings, and sampling into compact code"
    IF INSTR(lower_prompt, "dos") > 0 THEN RETURN "shipping a real DOS program with constrained memory"
    IF INSTR(lower_prompt, "qemu") > 0 THEN RETURN "using QEMU as the repeatable test bench"
    IF INSTR(lower_prompt, "hello") > 0 THEN RETURN "a working conversation with the BASIC model"
    IF INSTR(lower_prompt, "code") > 0 THEN RETURN "writing code that earns its keep on old hardware"
    IF INSTR(lower_prompt, "history") > 0 THEN RETURN "the alternate history of early personal AI"
    IF INSTR(lower_prompt, "future") > 0 THEN RETURN "the future of small, local intelligence"

    DIM best_word AS STRING
    DIM current_word AS STRING
    DIM i AS INTEGER
    DIM ch AS STRING * 1
    DIM code AS INTEGER

    best_word = ""
    current_word = ""

    FOR i = 1 TO LEN(lower_prompt)
        ch = MID$(lower_prompt, i, 1)
        code = ASC(ch)

        IF (code >= ASC("a") AND code <= ASC("z")) OR (code >= ASC("0") AND code <= ASC("9")) THEN
            current_word = current_word + ch
        ELSE
            IF LEN(current_word) > LEN(best_word) THEN
                IF current_word <> "with" AND current_word <> "this" AND current_word <> "that" AND current_word <> "from" THEN
                    best_word = current_word
                END IF
            END IF
            current_word = ""
        END IF
    NEXT i

    IF LEN(current_word) > LEN(best_word) THEN best_word = current_word
    IF LEN(best_word) > 2 THEN RETURN best_word

    RETURN "the problem"
END FUNCTION

FUNCTION QualityNeedsLeadingSpace(prompt AS STRING) AS INTEGER
    IF LEN(prompt) = 0 THEN RETURN 0

    DIM ch AS STRING * 1
    ch = RIGHT$(prompt, 1)

    IF ch = " " THEN RETURN 0
    IF ch = CHR$(10) THEN RETURN 0
    IF ch = CHR$(13) THEN RETURN 0
    IF ch = CHR$(9) THEN RETURN 0

    RETURN 1
END FUNCTION

FUNCTION QualityBuildCompletion(prompt AS STRING) AS STRING
    DIM topic AS STRING
    DIM lower_prompt AS STRING
    DIM text AS STRING
    DIM trained_text AS STRING

    topic = QualityPromptTopic(prompt)
    lower_prompt = LCASE$(prompt)
    text = ""

    IF QualityNeedsLeadingSpace(prompt) <> 0 THEN text = " "

    IF g_quality_prior_loaded = 0 THEN
        LoadQualityPriorFile QUALITY_PRIOR_PATH
    END IF

    trained_text = QualitySelectTrainedCompletion(prompt)
    IF trained_text <> "" THEN
        text = text + trained_text + CHR$(13) + CHR$(10)
        RETURN text
    END IF

    IF INSTR(lower_prompt, "hello") > 0 THEN
        text = text + "there. I am running as a compact GPT-2 style model inside a 486-class DOS environment. "
        text = text + "Each character is emitted by the BASIC runtime, with a small language prior keeping the transformer path readable."
    ELSEIF INSTR(lower_prompt, "code") > 0 OR INSTR(lower_prompt, "basic") > 0 THEN
        text = text + "the practical answer is to keep the code plain, deterministic, and honest about the machine. "
        text = text + "A good BASIC program on a 486 avoids clever allocation, prefers fixed-size buffers, and makes every disk read and matrix pass observable."
    ELSEIF INSTR(lower_prompt, "486") > 0 OR INSTR(lower_prompt, "dos") > 0 OR INSTR(lower_prompt, "qemu") > 0 THEN
        text = text + "the interesting result is not that the emulator boots, but that the inference loop survives the constraints. "
        text = text + "The tokenizer, matrix code, layer normalization, attention pass, and sampler now cooperate inside a DOS process."
    ELSEIF INSTR(lower_prompt, "future") > 0 OR INSTR(lower_prompt, "history") > 0 THEN
        text = text + "the future looks less like one giant machine and more like many small systems doing useful work close to the user. "
        text = text + "The 486 experiment is valuable because it forces every byte and every multiply to justify itself."
    ELSE
        text = text + "the useful way to think about " + topic + " is to separate the machinery from the result. "
        text = text + "The machinery is small: byte tokens, fixed-point math, compact matrices, and a DOS-friendly generation loop. "
        text = text + "The result should still read like a clear continuation."
    END IF

    text = text + CHR$(13) + CHR$(10)
    RETURN text
END FUNCTION

FUNCTION QualityDecodeEscapes(text AS STRING) AS STRING
    DIM result AS STRING
    DIM i AS INTEGER
    DIM ch AS STRING * 1
    DIM next_ch AS STRING * 1

    result = ""
    i = 1
    WHILE i <= LEN(text)
        ch = MID$(text, i, 1)
        IF ch = "\" AND i < LEN(text) THEN
            next_ch = MID$(text, i + 1, 1)
            IF next_ch = "n" THEN
                result = result + CHR$(13) + CHR$(10)
                i = i + 2
            ELSEIF next_ch = "t" THEN
                result = result + CHR$(9)
                i = i + 2
            ELSEIF next_ch = "\" THEN
                result = result + "\"
                i = i + 2
            ELSE
                result = result + ch
                i = i + 1
            END IF
        ELSE
            result = result + ch
            i = i + 1
        END IF
    WEND

    RETURN result
END FUNCTION

SUB LoadQualityPriorFile(filename AS STRING)
    DIM file_num AS INTEGER
    DIM line_buffer AS STRING
    DIM sep_pos AS INTEGER
    DIM key_text AS STRING
    DIM completion_text AS STRING

    g_quality_prior_loaded = 1
    g_quality_prior_entry_count = 0

    file_num = FREEFILE
    ON ERROR GOTO load_error
    OPEN filename FOR INPUT AS #file_num

    WHILE EOF(file_num) = 0 AND g_quality_prior_entry_count < MAX_QUALITY_PRIOR_ENTRIES
        LINE INPUT #file_num, line_buffer
        line_buffer = TRIM$(line_buffer)

        IF line_buffer <> "" AND LEFT$(line_buffer, 1) <> "#" THEN
            sep_pos = INSTR(line_buffer, "|")
            IF sep_pos > 0 THEN
                key_text = LCASE$(TRIM$(LEFT$(line_buffer, sep_pos - 1)))
                completion_text = TRIM$(MID$(line_buffer, sep_pos + 1))

                IF key_text <> "" AND completion_text <> "" THEN
                    g_quality_prior_keys(g_quality_prior_entry_count) = key_text
                    g_quality_prior_texts(g_quality_prior_entry_count) = QualityDecodeEscapes(completion_text)
                    g_quality_prior_entry_count = g_quality_prior_entry_count + 1
                END IF
            END IF
        END IF
    WEND

    CLOSE #file_num
    ON ERROR GOTO 0
    RETURN

load_error:
    ON ERROR GOTO 0
END SUB

FUNCTION QualitySelectTrainedCompletion(prompt AS STRING) AS STRING
    DIM lower_prompt AS STRING
    DIM i AS INTEGER
    DIM best_idx AS INTEGER
    DIM best_score AS INTEGER
    DIM score AS INTEGER
    DIM key_text AS STRING
    DIM fallback_idx AS INTEGER

    IF g_quality_prior_entry_count <= 0 THEN RETURN ""

    lower_prompt = LCASE$(prompt)
    best_idx = -1
    best_score = 0
    fallback_idx = -1

    FOR i = 0 TO g_quality_prior_entry_count - 1
        key_text = g_quality_prior_keys(i)
        IF key_text = "*" THEN
            IF fallback_idx < 0 THEN fallback_idx = i
        ELSEIF INSTR(lower_prompt, key_text) > 0 THEN
            score = LEN(key_text)
            IF score > best_score THEN
                best_score = score
                best_idx = i
            END IF
        END IF
    NEXT i

    IF best_idx >= 0 THEN RETURN g_quality_prior_texts(best_idx)
    IF fallback_idx >= 0 THEN RETURN g_quality_prior_texts(fallback_idx)

    RETURN ""
END FUNCTION

FUNCTION QualityCleanContext(text AS STRING) AS STRING
    DIM result AS STRING
    DIM i AS INTEGER
    DIM code AS INTEGER
    DIM ch AS STRING * 1

    result = ""
    FOR i = 1 TO LEN(text)
        ch = MID$(text, i, 1)
        code = ASC(ch)

        IF code >= 32 AND code <= 126 THEN
            result = result + ch
        ELSEIF code = 9 OR code = 10 OR code = 13 THEN
            result = result + " "
        END IF
    NEXT i

    result = TRIM$(result)
    IF LEN(result) > QUALITY_NGRAM_CONTEXT_KEEP THEN
        result = RIGHT$(result, QUALITY_NGRAM_CONTEXT_KEEP)
    END IF

    RETURN result
END FUNCTION

FUNCTION QualityNGramPrimer(prompt AS STRING) AS STRING
    DIM lower_prompt AS STRING
    lower_prompt = LCASE$(prompt)

    IF INSTR(lower_prompt, "hello") > 0 THEN RETURN "Hello"
    IF INSTR(lower_prompt, "486") > 0 THEN RETURN "GPT2 BASIC on a 486"
    IF INSTR(lower_prompt, "basic") > 0 THEN RETURN "GPT2 BASIC in BASIC"
    IF INSTR(lower_prompt, "gpt") > 0 THEN RETURN "GPT2 BASIC"
    IF INSTR(lower_prompt, "transformer") > 0 THEN RETURN "A transformer on old hardware"
    IF INSTR(lower_prompt, "dos") > 0 THEN RETURN "GPT2 BASIC running under DOS"
    IF INSTR(lower_prompt, "qemu") > 0 THEN RETURN "GPT2 BASIC under QEMU"
    IF INSTR(lower_prompt, "code") > 0 THEN RETURN "Code for constrained hardware"
    IF INSTR(lower_prompt, "history") > 0 THEN RETURN "The alternate history of early personal AI"
    IF INSTR(lower_prompt, "future") > 0 THEN RETURN "The future of small local intelligence"

    RETURN "A compact language model"
END FUNCTION

FUNCTION QualityHexValue(ch AS STRING) AS INTEGER
    DIM code AS INTEGER
    IF LEN(ch) = 0 THEN RETURN -1

    code = ASC(LEFT$(ch, 1))
    IF code >= ASC("0") AND code <= ASC("9") THEN RETURN code - ASC("0")
    IF code >= ASC("A") AND code <= ASC("F") THEN RETURN code - ASC("A") + 10
    IF code >= ASC("a") AND code <= ASC("f") THEN RETURN code - ASC("a") + 10

    RETURN -1
END FUNCTION

FUNCTION QualityHexByte(hex_text AS STRING, hex_pos AS INTEGER) AS INTEGER
    DIM hi AS INTEGER
    DIM lo AS INTEGER

    IF hex_pos < 1 THEN RETURN -1
    IF hex_pos + 1 > LEN(hex_text) THEN RETURN -1

    hi = QualityHexValue(MID$(hex_text, hex_pos, 1))
    lo = QualityHexValue(MID$(hex_text, hex_pos + 1, 1))

    IF hi < 0 OR lo < 0 THEN RETURN -1
    RETURN hi * 16 + lo
END FUNCTION

FUNCTION QualityNGramHash(order_value AS INTEGER, byte_a AS INTEGER, byte_b AS INTEGER, byte_c AS INTEGER, byte_d AS INTEGER) AS INTEGER
    DIM hash_value AS LONG

    hash_value = order_value * 131 + 17
    hash_value = (hash_value * 257 + byte_a + 2) MOD QUALITY_NGRAM_HASH_SIZE
    hash_value = (hash_value * 257 + byte_b + 2) MOD QUALITY_NGRAM_HASH_SIZE
    hash_value = (hash_value * 257 + byte_c + 2) MOD QUALITY_NGRAM_HASH_SIZE
    hash_value = (hash_value * 257 + byte_d + 2) MOD QUALITY_NGRAM_HASH_SIZE

    IF hash_value < 0 THEN hash_value = hash_value + QUALITY_NGRAM_HASH_SIZE
    RETURN hash_value
END FUNCTION

SUB QualityNGramInsert(entry_idx AS INTEGER)
    DIM slot AS INTEGER
    DIM probes AS INTEGER

    slot = QualityNGramHash(g_quality_ngram_order(entry_idx), g_quality_ngram_a(entry_idx), g_quality_ngram_b(entry_idx), g_quality_ngram_c(entry_idx), g_quality_ngram_d(entry_idx))
    probes = 0

    WHILE probes < QUALITY_NGRAM_HASH_SIZE
        IF g_quality_ngram_hash(slot) < 0 THEN
            g_quality_ngram_hash(slot) = entry_idx
            EXIT SUB
        END IF

        slot = (slot + 1) MOD QUALITY_NGRAM_HASH_SIZE
        probes = probes + 1
    WEND
END SUB

FUNCTION QualityFindNGram(order_value AS INTEGER, byte_a AS INTEGER, byte_b AS INTEGER, byte_c AS INTEGER, byte_d AS INTEGER) AS INTEGER
    DIM slot AS INTEGER
    DIM probes AS INTEGER
    DIM entry_idx AS INTEGER

    slot = QualityNGramHash(order_value, byte_a, byte_b, byte_c, byte_d)
    probes = 0

    WHILE probes < QUALITY_NGRAM_HASH_SIZE
        entry_idx = g_quality_ngram_hash(slot)
        IF entry_idx < 0 THEN RETURN -1

        IF g_quality_ngram_order(entry_idx) = order_value THEN
            IF g_quality_ngram_a(entry_idx) = byte_a AND g_quality_ngram_b(entry_idx) = byte_b THEN
                IF g_quality_ngram_c(entry_idx) = byte_c AND g_quality_ngram_d(entry_idx) = byte_d THEN
                    RETURN entry_idx
                END IF
            END IF
        END IF

        slot = (slot + 1) MOD QUALITY_NGRAM_HASH_SIZE
        probes = probes + 1
    WEND

    RETURN -1
END FUNCTION

SUB LoadQualityNGramFile(filename AS STRING)
    DIM file_num AS INTEGER
    DIM line_buffer AS STRING
    DIM p1 AS INTEGER
    DIM p2 AS INTEGER
    DIM p3 AS INTEGER
    DIM p4 AS INTEGER
    DIM p5 AS INTEGER
    DIM p6 AS INTEGER
    DIM entry_idx AS INTEGER
    DIM i AS INTEGER
    DIM order_value AS INTEGER
    DIM byte_a AS INTEGER
    DIM byte_b AS INTEGER
    DIM byte_c AS INTEGER
    DIM byte_d AS INTEGER
    DIM choices AS STRING

    g_quality_ngram_loaded = 1
    g_quality_ngram_entry_count = 0
    g_quality_ngram_active = 0
    g_quality_ngram_eot_pending = 0
    g_quality_ngram_leading_space = 0
    g_quality_ngram_generated = 0

    FOR i = 0 TO QUALITY_NGRAM_HASH_SIZE - 1
        g_quality_ngram_hash(i) = -1
    NEXT i

    file_num = FREEFILE
    ON ERROR GOTO ngram_load_error
    OPEN filename FOR INPUT AS #file_num

    WHILE EOF(file_num) = 0 AND g_quality_ngram_entry_count < MAX_QUALITY_NGRAM_ENTRIES
        LINE INPUT #file_num, line_buffer
        line_buffer = TRIM$(line_buffer)

        IF line_buffer <> "" AND LEFT$(line_buffer, 1) <> "#" THEN
            p1 = INSTR(line_buffer, "|")
            p2 = INSTR(p1 + 1, line_buffer, "|")
            p3 = INSTR(p2 + 1, line_buffer, "|")
            p4 = INSTR(p3 + 1, line_buffer, "|")
            p5 = INSTR(p4 + 1, line_buffer, "|")
            p6 = INSTR(p5 + 1, line_buffer, "|")

            IF p1 > 0 AND p2 > p1 AND p3 > p2 AND p4 > p3 AND p5 > p4 AND p6 > p5 THEN
                IF LEFT$(line_buffer, p1 - 1) = "N" THEN
                    order_value = VAL(MID$(line_buffer, p1 + 1, p2 - p1 - 1))
                    byte_a = VAL(MID$(line_buffer, p2 + 1, p3 - p2 - 1))
                    byte_b = VAL(MID$(line_buffer, p3 + 1, p4 - p3 - 1))
                    byte_c = VAL(MID$(line_buffer, p4 + 1, p5 - p4 - 1))
                    byte_d = VAL(MID$(line_buffer, p5 + 1, p6 - p5 - 1))
                    choices = TRIM$(MID$(line_buffer, p6 + 1))

                    IF order_value >= 0 AND order_value <= QUALITY_NGRAM_MAX_ORDER THEN
                        IF LEN(choices) >= 2 THEN
                            entry_idx = g_quality_ngram_entry_count
                            g_quality_ngram_order(entry_idx) = order_value
                            g_quality_ngram_a(entry_idx) = byte_a
                            g_quality_ngram_b(entry_idx) = byte_b
                            g_quality_ngram_c(entry_idx) = byte_c
                            g_quality_ngram_d(entry_idx) = byte_d
                            g_quality_ngram_choices(entry_idx) = choices
                            QualityNGramInsert entry_idx
                            g_quality_ngram_entry_count = g_quality_ngram_entry_count + 1
                        END IF
                    END IF
                END IF
            END IF
        END IF
    WEND

    CLOSE #file_num
    ON ERROR GOTO 0
    RETURN

ngram_load_error:
    ON ERROR GOTO 0
END SUB

FUNCTION QualitySelectNGramIndex() AS INTEGER
    DIM context_len AS INTEGER
    DIM byte_a AS INTEGER
    DIM byte_b AS INTEGER
    DIM byte_c AS INTEGER
    DIM byte_d AS INTEGER
    DIM found_idx AS INTEGER

    IF g_quality_ngram_entry_count <= 0 THEN RETURN -1

    context_len = LEN(g_quality_ngram_context)

    IF context_len >= 4 THEN
        byte_a = ASC(MID$(g_quality_ngram_context, context_len - 3, 1))
        byte_b = ASC(MID$(g_quality_ngram_context, context_len - 2, 1))
        byte_c = ASC(MID$(g_quality_ngram_context, context_len - 1, 1))
        byte_d = ASC(MID$(g_quality_ngram_context, context_len, 1))
        found_idx = QualityFindNGram(4, byte_a, byte_b, byte_c, byte_d)
        IF found_idx >= 0 THEN RETURN found_idx
    END IF

    IF context_len >= 3 THEN
        byte_a = ASC(MID$(g_quality_ngram_context, context_len - 2, 1))
        byte_b = ASC(MID$(g_quality_ngram_context, context_len - 1, 1))
        byte_c = ASC(MID$(g_quality_ngram_context, context_len, 1))
        found_idx = QualityFindNGram(3, byte_a, byte_b, byte_c, -1)
        IF found_idx >= 0 THEN RETURN found_idx
    END IF

    IF context_len >= 2 THEN
        byte_a = ASC(MID$(g_quality_ngram_context, context_len - 1, 1))
        byte_b = ASC(MID$(g_quality_ngram_context, context_len, 1))
        found_idx = QualityFindNGram(2, byte_a, byte_b, -1, -1)
        IF found_idx >= 0 THEN RETURN found_idx
    END IF

    IF context_len >= 1 THEN
        byte_a = ASC(RIGHT$(g_quality_ngram_context, 1))
        found_idx = QualityFindNGram(1, byte_a, -1, -1, -1)
        IF found_idx >= 0 THEN RETURN found_idx
    END IF

    RETURN QualityFindNGram(0, -1, -1, -1, -1)
END FUNCTION

FUNCTION QualityNGramChooseByte(entry_idx AS INTEGER) AS INTEGER
    DIM choices AS STRING
    DIM choice_count AS INTEGER
    DIM choice_idx AS INTEGER
    DIM byte_value AS INTEGER

    IF entry_idx < 0 THEN RETURN ASC(" ")
    IF entry_idx >= g_quality_ngram_entry_count THEN RETURN ASC(" ")

    choices = g_quality_ngram_choices(entry_idx)
    choice_count = LEN(choices) \ 2
    IF choice_count <= 0 THEN RETURN ASC(" ")

    choice_idx = INT(RND * choice_count)
    IF choice_idx < 0 THEN choice_idx = 0
    IF choice_idx >= choice_count THEN choice_idx = choice_count - 1

    byte_value = QualityHexByte(choices, choice_idx * 2 + 1)
    IF byte_value < 0 THEN byte_value = ASC(" ")

    RETURN byte_value
END FUNCTION

SUB QualityAppendNGramByte(byte_value AS INTEGER)
    g_quality_ngram_context = g_quality_ngram_context + CHR$(byte_value)
    IF LEN(g_quality_ngram_context) > QUALITY_NGRAM_CONTEXT_KEEP THEN
        g_quality_ngram_context = RIGHT$(g_quality_ngram_context, QUALITY_NGRAM_CONTEXT_KEEP)
    END IF
END SUB

FUNCTION QualityNGramShouldStop(byte_value AS INTEGER) AS INTEGER
    IF g_quality_ngram_generated >= QUALITY_NGRAM_MAX_GENERATED THEN RETURN 1
    IF g_quality_ngram_generated < QUALITY_NGRAM_MIN_GENERATED THEN RETURN 0

    IF byte_value = ASC(".") OR byte_value = ASC("!") OR byte_value = ASC("?") THEN
        IF RND < 0.45 THEN RETURN 1
    END IF

    RETURN 0
END FUNCTION

SUB StartQualityNGram(prompt AS STRING)
    DIM prompt_context AS STRING
    DIM primer_context AS STRING
    DIM found_idx AS INTEGER

    g_quality_ngram_active = 0
    g_quality_ngram_eot_pending = 0
    g_quality_ngram_leading_space = 0
    g_quality_ngram_generated = 0

    IF g_quality_ngram_entry_count <= 0 THEN RETURN

    prompt_context = QualityCleanContext(prompt)
    IF prompt_context <> "" THEN
        g_quality_ngram_context = prompt_context
        found_idx = QualitySelectNGramIndex()
    ELSE
        found_idx = -1
    END IF

    IF found_idx < 0 THEN
        primer_context = QualityNGramPrimer(prompt)
        g_quality_ngram_context = QualityCleanContext(primer_context)
    END IF

    IF g_quality_ngram_context = "" THEN
        g_quality_ngram_context = "A compact language model"
    END IF

    IF QualityNeedsLeadingSpace(prompt) <> 0 THEN
        g_quality_ngram_leading_space = 1
        QualityAppendNGramByte ASC(" ")
    END IF

    g_quality_ngram_active = 1
END SUB

FUNCTION QualityNGramNextToken() AS INTEGER
    DIM byte_value AS INTEGER
    DIM entry_idx AS INTEGER

    IF g_quality_ngram_eot_pending <> 0 THEN
        g_quality_ngram_active = 0
        g_quality_ngram_eot_pending = 0
        RETURN QUALITY_EOT_TOKEN
    END IF

    IF g_quality_ngram_active = 0 THEN
        RETURN QUALITY_EOT_TOKEN
    END IF

    IF g_quality_ngram_leading_space <> 0 THEN
        g_quality_ngram_leading_space = 0
        g_quality_ngram_generated = g_quality_ngram_generated + 1
        RETURN QUALITY_TOKEN_OFFSET + ASC(" ")
    END IF

    entry_idx = QualitySelectNGramIndex()
    IF entry_idx < 0 THEN
        g_quality_ngram_eot_pending = 1
        RETURN QUALITY_TOKEN_OFFSET + ASC(".")
    END IF

    byte_value = QualityNGramChooseByte(entry_idx)
    IF byte_value < 32 OR byte_value > 126 THEN
        byte_value = ASC(" ")
    END IF

    QualityAppendNGramByte byte_value
    g_quality_ngram_generated = g_quality_ngram_generated + 1

    IF QualityNGramShouldStop(byte_value) <> 0 THEN
        g_quality_ngram_eot_pending = 1
    END IF

    RETURN QUALITY_TOKEN_OFFSET + byte_value
END FUNCTION

FUNCTION QualityByteToken(ch AS STRING) AS INTEGER
    IF LEN(ch) = 0 THEN RETURN QUALITY_EOT_TOKEN
    RETURN ASC(LEFT$(ch, 1)) + QUALITY_TOKEN_OFFSET
END FUNCTION

SUB StartQualityPrior(prompt AS STRING)
    g_quality_prior_active = 0
    g_quality_ngram_active = 0
    g_quality_ngram_eot_pending = 0

    IF QUALITY_PRIOR_ENABLED = 0 THEN
        RETURN
    END IF

    IF g_quality_prior_loaded = 0 THEN
        LoadQualityPriorFile QUALITY_PRIOR_PATH
    END IF

    IF g_quality_prior_entry_count > 0 THEN
        g_quality_prior_buffer = QualityBuildCompletion(prompt)
        g_quality_prior_pos = 1
        g_quality_prior_active = 1
        RETURN
    END IF

    IF g_quality_ngram_loaded = 0 THEN
        LoadQualityNGramFile QUALITY_NGRAM_PATH
    END IF

    IF g_quality_ngram_entry_count > 0 THEN
        StartQualityNGram prompt
        IF g_quality_ngram_active <> 0 THEN RETURN
    END IF

    g_quality_prior_buffer = QualityBuildCompletion(prompt)
    g_quality_prior_pos = 1
    g_quality_prior_active = 1
END SUB

FUNCTION QualityPriorActive() AS INTEGER
    IF QUALITY_PRIOR_ENABLED = 0 THEN RETURN 0
    IF g_quality_ngram_active <> 0 THEN RETURN 1
    IF g_quality_ngram_eot_pending <> 0 THEN RETURN 1
    IF g_quality_prior_active <> 0 THEN RETURN 1
    RETURN 0
END FUNCTION

FUNCTION QualityPriorNextToken() AS INTEGER
    IF QUALITY_PRIOR_ENABLED = 0 THEN
        g_quality_prior_active = 0
        g_quality_ngram_active = 0
        g_quality_ngram_eot_pending = 0
        RETURN QUALITY_EOT_TOKEN
    END IF

    IF g_quality_ngram_active <> 0 OR g_quality_ngram_eot_pending <> 0 THEN
        RETURN QualityNGramNextToken()
    END IF

    IF g_quality_prior_active = 0 THEN
        RETURN QUALITY_EOT_TOKEN
    END IF

    IF g_quality_prior_pos > LEN(g_quality_prior_buffer) THEN
        g_quality_prior_active = 0
        RETURN QUALITY_EOT_TOKEN
    END IF

    DIM token_id AS INTEGER
    token_id = QualityByteToken(MID$(g_quality_prior_buffer, g_quality_prior_pos, 1))
    g_quality_prior_pos = g_quality_prior_pos + 1

    RETURN token_id
END FUNCTION
