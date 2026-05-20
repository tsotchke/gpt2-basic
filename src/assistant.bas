' *******************************************************
' * GPT2-BASIC Assistant Pack Shell                    *
' *******************************************************
' * Optional Clippy-style assistant surface. This file  *
' * builds ASSIST.EXE, separate from the slim release   *
' * GPT2.EXE. It loads assistant packs, switches model  *
' * paths, retrieves pack notes, and renders a text UI  *
' * that can later be skinned with VGA sprites/icons.   *
' *******************************************************

#INCLUDE "src/tokenizer.bas"

DIM SHARED g_assist_current_memory AS LONG
DIM SHARED g_assist_peak_memory AS LONG

DECLARE SUB TrackAllocation(size AS LONG)
DECLARE SUB TrackDeallocation(size AS LONG)

SUB TrackAllocation(size AS LONG)
    IF size <= 0 THEN RETURN
    g_assist_current_memory = g_assist_current_memory + size
    IF g_assist_current_memory > g_assist_peak_memory THEN g_assist_peak_memory = g_assist_current_memory
END SUB

SUB TrackDeallocation(size AS LONG)
    IF size <= 0 THEN RETURN
    g_assist_current_memory = g_assist_current_memory - size
    IF g_assist_current_memory < 0 THEN g_assist_current_memory = 0
END SUB

#INCLUDE "src/real_gpt.bas"

CONST ASSIST_PACK_ROOT = "PACKS"
CONST ASSIST_PACK_LIST = "PACKS\PACKS.TXT"
CONST ASSIST_MAX_PACKS = 8
CONST ASSIST_MAX_REPLY_TOKENS = 64
CONST ASSIST_SENTENCE_STOP_MIN_TOKENS = 10
CONST ASSIST_DEFAULT_TOP_P = 0.9
CONST ASSIST_DEFAULT_TOP_K = 24
CONST ASSIST_HISTORY_MAX = 96
CONST ASSIST_HISTORY_PAGE = 10
CONST ASSIST_MEMORY_VALUE_MAX = 72

TYPE AssistantPack
    id AS STRING * 32
    title AS STRING * 64
    model_path AS STRING * 80
    persona AS STRING * 160
    help_path AS STRING * 80
    golden_path AS STRING * 80
    usage_path AS STRING * 80
    sprite_path AS STRING * 80
    icons_path AS STRING * 80
    actions AS STRING * 80
    loaded AS INTEGER
END TYPE

DIM SHARED g_assist_packs(0 TO ASSIST_MAX_PACKS - 1) AS AssistantPack
DIM SHARED g_assist_pack_count AS INTEGER
DIM SHARED g_assist_active_pack AS INTEGER
DIM SHARED g_assist_model_path AS STRING
DIM SHARED g_assist_emit_records AS INTEGER
DIM SHARED g_assist_history(0 TO ASSIST_HISTORY_MAX - 1) AS STRING
DIM SHARED g_assist_history_count AS INTEGER
DIM SHARED g_assist_history_scroll AS INTEGER
DIM SHARED g_assist_memory_user_name AS STRING
DIM SHARED g_assist_memory_goal AS STRING
DIM SHARED g_assist_memory_style AS STRING
DIM SHARED g_assist_memory_problem AS STRING
DIM SHARED g_assist_memory_last_user AS STRING
DIM SHARED g_assist_memory_last_answer AS STRING

DECLARE FUNCTION AssistTrimFixed(value AS STRING) AS STRING
DECLARE FUNCTION AssistPathJoin(left_path AS STRING, right_path AS STRING) AS STRING
DECLARE FUNCTION AssistSafeText(value AS STRING) AS STRING
DECLARE FUNCTION AssistVisibleToken(token_id AS INTEGER) AS STRING
DECLARE SUB AssistAddHistory(line_text AS STRING)
DECLARE SUB AssistClearHistory()
DECLARE SUB AssistRenderHistory()
DECLARE SUB AssistHistoryPage(delta AS INTEGER)
DECLARE FUNCTION AssistCleanMemoryValue(value AS STRING) AS STRING
DECLARE FUNCTION AssistMemoryExtract(original AS STRING, lower_text AS STRING, marker AS STRING) AS STRING
DECLARE SUB AssistRememberFact(key_name AS STRING, value AS STRING)
DECLARE SUB AssistRememberTurn(query AS STRING, answer AS STRING)
DECLARE FUNCTION AssistMemoryFactsText() AS STRING
DECLARE FUNCTION AssistMemoryContext() AS STRING
DECLARE FUNCTION AssistMemoryReply(query AS STRING) AS STRING
DECLARE FUNCTION AssistRememberCommand(argument AS STRING) AS STRING
DECLARE SUB AssistClearMemoryFacts()
DECLARE SUB AssistPrintMemory()
DECLARE FUNCTION AssistReadIniValue(filename AS STRING, key_name AS STRING, default_value AS STRING) AS STRING
DECLARE FUNCTION AssistLoadPack(pack_id AS STRING, pack_index AS INTEGER) AS INTEGER
DECLARE SUB AssistLoadPackList()
DECLARE SUB AssistUseBuiltinPack()
DECLARE SUB AssistPrintPackList()
DECLARE FUNCTION AssistSelectPack(pack_id AS STRING) AS INTEGER
DECLARE FUNCTION AssistTokenizerModeName(mode_value AS INTEGER) AS STRING
DECLARE FUNCTION AssistInitializeModel(pack_index AS INTEGER) AS INTEGER
DECLARE SUB AssistPreloadActivePackModel()
DECLARE SUB AssistShutdownModel()
DECLARE FUNCTION AssistClassifyIntent(query AS STRING) AS STRING
DECLARE FUNCTION AssistActionsForIntent(intent_name AS STRING) AS STRING
DECLARE FUNCTION AssistRetrieve(pack_index AS INTEGER, query AS STRING) AS STRING
DECLARE FUNCTION AssistGoldenReply(pack_index AS INTEGER, query AS STRING) AS STRING
DECLARE FUNCTION AssistGenerate(prompt AS STRING, max_tokens AS INTEGER) AS STRING
DECLARE FUNCTION AssistStreamGenerate(prompt AS STRING, max_tokens AS INTEGER) AS STRING
DECLARE FUNCTION AssistCleanGeneratedText(raw_text AS STRING) AS STRING
DECLARE FUNCTION AssistTextHasRepeatedChunk(value AS STRING) AS INTEGER
DECLARE FUNCTION AssistGeneratedLooksBad(generated AS STRING, query AS STRING) AS INTEGER
DECLARE FUNCTION AssistFallbackReply(pack_index AS INTEGER, intent_name AS STRING, query AS STRING, retrieved AS STRING) AS STRING
DECLARE SUB AssistPrepareGenerationPrompt(input_tokens() AS INTEGER, BYREF input_count AS INTEGER, max_tokens AS INTEGER)
DECLARE SUB AssistPrefillPrompt(input_tokens() AS INTEGER, input_count AS INTEGER)
DECLARE SUB AssistRenderFrame()
DECLARE SUB AssistRenderPackStatus()
DECLARE SUB AssistPrintPackUsage(pack_index AS INTEGER)
DECLARE SUB AssistRenderReply(query AS STRING, use_generation AS INTEGER)
DECLARE SUB AssistScriptedDemo()
DECLARE SUB AssistGuardProbe()
DECLARE SUB AssistStressProbe()
DECLARE SUB AssistInteractive()
DECLARE SUB AssistMain()

FUNCTION AssistTrimFixed(value AS STRING) AS STRING
    RETURN TRIM$(RTRIM$(value))
END FUNCTION

FUNCTION AssistPathJoin(left_path AS STRING, right_path AS STRING) AS STRING
    IF left_path = "" THEN RETURN right_path
    IF right_path = "" THEN RETURN left_path
    IF RIGHT$(left_path, 1) = "\" THEN RETURN left_path + right_path
    RETURN left_path + "\" + right_path
END FUNCTION

FUNCTION AssistSafeText(value AS STRING) AS STRING
    DIM result AS STRING
    DIM i AS INTEGER
    DIM ch AS STRING
    DIM code AS INTEGER

    result = ""
    FOR i = 1 TO LEN(value)
        ch = MID$(value, i, 1)
        code = ASC(ch)
        IF ch = "|" THEN
            result = result + "/"
        ELSEIF code = 13 OR code = 10 OR code = 9 THEN
            result = result + " "
        ELSEIF code < 32 OR code > 126 THEN
            result = result + "."
        ELSE
            result = result + ch
        END IF
    NEXT i

    RETURN result
END FUNCTION

FUNCTION AssistVisibleToken(token_id AS INTEGER) AS STRING
    DIM raw_text AS STRING
    DIM result AS STRING
    DIM i AS INTEGER
    DIM code AS INTEGER
    DIM ch AS STRING

    IF token_id = 0 THEN RETURN "<eot>"
    raw_text = TinyGPTTokenText(token_id)
    IF raw_text = "" THEN RETURN "<tok" + LTRIM$(STR$(token_id)) + ">"

    result = ""
    FOR i = 1 TO LEN(raw_text)
        ch = MID$(raw_text, i, 1)
        code = ASC(ch)
        IF code = 32 THEN
            result = result + "_"
        ELSEIF code < 32 OR code > 126 THEN
            result = result + "."
        ELSE
            result = result + ch
        END IF
        IF LEN(result) >= 12 THEN EXIT FOR
    NEXT i

    IF result = "" THEN result = "<blank>"
    IF LEN(raw_text) > 12 THEN result = result + "+"
    RETURN result
END FUNCTION

SUB AssistAddHistory(line_text AS STRING)
    DIM i AS INTEGER

    line_text = TRIM$(line_text)
    IF line_text = "" THEN RETURN
    IF LEN(line_text) > 360 THEN line_text = LEFT$(line_text, 360)

    IF g_assist_history_count < ASSIST_HISTORY_MAX THEN
        g_assist_history(g_assist_history_count) = line_text
        g_assist_history_count = g_assist_history_count + 1
    ELSE
        FOR i = 0 TO ASSIST_HISTORY_MAX - 2
            g_assist_history(i) = g_assist_history(i + 1)
        NEXT i
        g_assist_history(ASSIST_HISTORY_MAX - 1) = line_text
    END IF

    g_assist_history_scroll = 0
END SUB

SUB AssistClearHistory()
    DIM i AS INTEGER
    FOR i = 0 TO ASSIST_HISTORY_MAX - 1
        g_assist_history(i) = ""
    NEXT i
    g_assist_history_count = 0
    g_assist_history_scroll = 0
END SUB

SUB AssistRenderHistory()
    DIM start_index AS INTEGER
    DIM end_index AS INTEGER
    DIM i AS INTEGER

    AssistRenderFrame
    AssistRenderPackStatus
    PRINT "Transcript: /u or /up, /d or /down, /home, /end, /clear, /packs, /pack NAME, /quit"
    PRINT

    IF g_assist_history_count = 0 THEN
        PRINT "(No transcript yet. Type a question at the prompt.)"
        PRINT
        RETURN
    END IF

    end_index = g_assist_history_count - 1 - g_assist_history_scroll
    IF end_index < 0 THEN end_index = 0
    start_index = end_index - ASSIST_HISTORY_PAGE + 1
    IF start_index < 0 THEN start_index = 0

    FOR i = start_index TO end_index
        PRINT g_assist_history(i)
        PRINT
    NEXT i

    PRINT "Showing "; start_index + 1; "-"; end_index + 1; " of "; g_assist_history_count; "."
    PRINT
END SUB

SUB AssistHistoryPage(delta AS INTEGER)
    DIM max_scroll AS INTEGER

    max_scroll = g_assist_history_count - ASSIST_HISTORY_PAGE
    IF max_scroll < 0 THEN max_scroll = 0

    g_assist_history_scroll = g_assist_history_scroll + delta
    IF g_assist_history_scroll < 0 THEN g_assist_history_scroll = 0
    IF g_assist_history_scroll > max_scroll THEN g_assist_history_scroll = max_scroll

    AssistRenderHistory
END SUB

FUNCTION AssistCleanMemoryValue(value AS STRING) AS STRING
    DIM result AS STRING

    result = AssistSafeText(TRIM$(value))
    WHILE LEN(result) > 0 AND (RIGHT$(result, 1) = "." OR RIGHT$(result, 1) = "!" OR RIGHT$(result, 1) = "?" OR RIGHT$(result, 1) = ",")
        result = LEFT$(result, LEN(result) - 1)
        result = RTRIM$(result)
    WEND
    IF LEN(result) > ASSIST_MEMORY_VALUE_MAX THEN result = LEFT$(result, ASSIST_MEMORY_VALUE_MAX)
    RETURN result
END FUNCTION

FUNCTION AssistMemoryExtract(original AS STRING, lower_text AS STRING, marker AS STRING) AS STRING
    DIM marker_pos AS INTEGER

    marker_pos = INSTR(lower_text, marker)
    IF marker_pos = 0 THEN RETURN ""
    RETURN AssistCleanMemoryValue(MID$(original, marker_pos + LEN(marker)))
END FUNCTION

SUB AssistRememberFact(key_name AS STRING, value AS STRING)
    DIM key_text AS STRING
    DIM clean_value AS STRING

    key_text = LCASE$(TRIM$(key_name))
    clean_value = AssistCleanMemoryValue(value)
    IF clean_value = "" THEN RETURN

    IF key_text = "name" OR key_text = "user" THEN
        g_assist_memory_user_name = clean_value
    ELSEIF key_text = "goal" OR key_text = "work" OR key_text = "topic" THEN
        g_assist_memory_goal = clean_value
    ELSEIF key_text = "style" OR key_text = "preference" THEN
        g_assist_memory_style = clean_value
    ELSEIF key_text = "problem" OR key_text = "issue" THEN
        g_assist_memory_problem = clean_value
    END IF
END SUB

SUB AssistRememberTurn(query AS STRING, answer AS STRING)
    g_assist_memory_last_user = AssistCleanMemoryValue(query)
    g_assist_memory_last_answer = AssistCleanMemoryValue(answer)
END SUB

FUNCTION AssistMemoryFactsText() AS STRING
    DIM facts AS STRING

    facts = ""
    IF g_assist_memory_user_name <> "" THEN facts = facts + "name=" + g_assist_memory_user_name + "; "
    IF g_assist_memory_goal <> "" THEN facts = facts + "goal=" + g_assist_memory_goal + "; "
    IF g_assist_memory_style <> "" THEN facts = facts + "style=" + g_assist_memory_style + "; "
    IF g_assist_memory_problem <> "" THEN facts = facts + "problem=" + g_assist_memory_problem + "; "
    IF g_assist_memory_last_user <> "" THEN facts = facts + "last_user=" + g_assist_memory_last_user + "; "
    IF facts = "" THEN RETURN "Memory is empty."
    IF RIGHT$(facts, 2) = "; " THEN facts = LEFT$(facts, LEN(facts) - 2)
    RETURN "Memory: " + facts + "."
END FUNCTION

FUNCTION AssistMemoryContext() AS STRING
    DIM context_text AS STRING

    context_text = ""
    IF g_assist_memory_user_name <> "" THEN context_text = context_text + "user name is " + g_assist_memory_user_name + "; "
    IF g_assist_memory_goal <> "" THEN context_text = context_text + "current goal is " + g_assist_memory_goal + "; "
    IF g_assist_memory_style <> "" THEN context_text = context_text + "answer style is " + g_assist_memory_style + "; "
    IF g_assist_memory_problem <> "" THEN context_text = context_text + "known problem is " + g_assist_memory_problem + "; "
    IF g_assist_memory_last_user <> "" THEN context_text = context_text + "previous question was " + g_assist_memory_last_user + "; "
    IF g_assist_memory_last_answer <> "" THEN context_text = context_text + "previous answer was " + g_assist_memory_last_answer + "; "
    IF context_text = "" THEN RETURN ""
    IF RIGHT$(context_text, 2) = "; " THEN context_text = LEFT$(context_text, LEN(context_text) - 2)
    RETURN "Context: " + context_text + "."
END FUNCTION

FUNCTION AssistMemoryReply(query AS STRING) AS STRING
    DIM lower_text AS STRING
    DIM value_text AS STRING

    lower_text = LCASE$(TRIM$(query))
    value_text = ""

    value_text = AssistMemoryExtract(query, lower_text, "my name is ")
    IF value_text <> "" THEN
        AssistRememberFact "name", value_text
        RETURN "I will remember your name is " + value_text + "."
    END IF

    value_text = AssistMemoryExtract(query, lower_text, "call me ")
    IF value_text <> "" THEN
        AssistRememberFact "name", value_text
        RETURN "I will remember your name is " + value_text + "."
    END IF

    value_text = AssistMemoryExtract(query, lower_text, "we are working on ")
    IF value_text = "" THEN value_text = AssistMemoryExtract(query, lower_text, "we're working on ")
    IF value_text = "" THEN value_text = AssistMemoryExtract(query, lower_text, "our goal is ")
    IF value_text = "" THEN value_text = AssistMemoryExtract(query, lower_text, "the goal is ")
    IF value_text <> "" THEN
        AssistRememberFact "goal", value_text
        RETURN "I will remember we are working on " + value_text + "."
    END IF

    value_text = AssistMemoryExtract(query, lower_text, "i prefer ")
    IF value_text = "" THEN value_text = AssistMemoryExtract(query, lower_text, "please keep answers ")
    IF value_text <> "" THEN
        AssistRememberFact "style", value_text
        RETURN "I will remember you prefer " + value_text + "."
    END IF

    value_text = AssistMemoryExtract(query, lower_text, "the problem is ")
    IF value_text = "" THEN value_text = AssistMemoryExtract(query, lower_text, "my problem is ")
    IF value_text <> "" THEN
        AssistRememberFact "problem", value_text
        RETURN "I will remember the problem is " + value_text + "."
    END IF

    IF INSTR(lower_text, "what is my name") > 0 OR INSTR(lower_text, "who am i") > 0 THEN
        IF g_assist_memory_user_name = "" THEN RETURN "I do not know your name yet."
        RETURN "Your name is " + g_assist_memory_user_name + "."
    END IF

    IF INSTR(lower_text, "what are we working on") > 0 OR INSTR(lower_text, "what is our goal") > 0 THEN
        IF g_assist_memory_goal = "" THEN RETURN "I do not know the current goal yet."
        RETURN "We are working on " + g_assist_memory_goal + "."
    END IF

    IF INSTR(lower_text, "how should you answer") > 0 OR INSTR(lower_text, "what style") > 0 THEN
        IF g_assist_memory_style = "" THEN RETURN "I do not know your answer style yet."
        RETURN "I should answer " + g_assist_memory_style + "."
    END IF

    IF INSTR(lower_text, "what was the problem") > 0 OR INSTR(lower_text, "what problem") > 0 THEN
        IF g_assist_memory_problem = "" THEN RETURN "I do not know the problem yet."
        RETURN "The problem is " + g_assist_memory_problem + "."
    END IF

    IF INSTR(lower_text, "what did i just ask") > 0 OR INSTR(lower_text, "what was my last question") > 0 THEN
        IF g_assist_memory_last_user = "" THEN RETURN "I do not have a previous question yet."
        RETURN "You just asked: " + g_assist_memory_last_user + "."
    END IF

    IF INSTR(lower_text, "what do you remember") > 0 OR INSTR(lower_text, "show memory") > 0 THEN
        RETURN AssistMemoryFactsText()
    END IF

    RETURN ""
END FUNCTION

FUNCTION AssistRememberCommand(argument AS STRING) AS STRING
    DIM eq_pos AS INTEGER
    DIM key_text AS STRING
    DIM value_text AS STRING
    DIM key_supported AS INTEGER

    eq_pos = INSTR(argument, "=")
    IF eq_pos <= 1 THEN RETURN "Use /remember name=value, goal=value, style=value, or problem=value."

    key_text = TRIM$(LEFT$(argument, eq_pos - 1))
    value_text = AssistCleanMemoryValue(MID$(argument, eq_pos + 1))
    key_supported = 0
    IF LCASE$(key_text) = "name" OR LCASE$(key_text) = "user" THEN key_supported = 1
    IF LCASE$(key_text) = "goal" OR LCASE$(key_text) = "work" OR LCASE$(key_text) = "topic" THEN key_supported = 1
    IF LCASE$(key_text) = "style" OR LCASE$(key_text) = "preference" THEN key_supported = 1
    IF LCASE$(key_text) = "problem" OR LCASE$(key_text) = "issue" THEN key_supported = 1
    IF key_supported = 0 THEN RETURN "Use /remember name=value, goal=value, style=value, or problem=value."
    AssistRememberFact key_text, value_text
    IF value_text = "" THEN RETURN "Use /remember name=value, goal=value, style=value, or problem=value."
    RETURN "I will remember " + LCASE$(key_text) + "=" + value_text + "."
END FUNCTION

SUB AssistClearMemoryFacts()
    g_assist_memory_user_name = ""
    g_assist_memory_goal = ""
    g_assist_memory_style = ""
    g_assist_memory_problem = ""
    g_assist_memory_last_user = ""
    g_assist_memory_last_answer = ""
END SUB

SUB AssistPrintMemory()
    PRINT AssistMemoryFactsText()
    PRINT
END SUB

FUNCTION AssistReadIniValue(filename AS STRING, key_name AS STRING, default_value AS STRING) AS STRING
    DIM file_num AS INTEGER
    DIM line_text AS STRING
    DIM eq_pos AS INTEGER
    DIM key_text AS STRING
    DIM value_text AS STRING
    DIM wanted_key AS STRING

    wanted_key = UCASE$(TRIM$(key_name))
    file_num = FREEFILE
    ON ERROR GOTO assist_ini_open_error
    OPEN filename FOR INPUT AS #file_num
    ON ERROR GOTO 0

    WHILE EOF(file_num) = 0
        LINE INPUT #file_num, line_text
        line_text = TRIM$(line_text)
        IF line_text <> "" AND LEFT$(line_text, 1) <> "#" AND LEFT$(line_text, 1) <> ";" THEN
            eq_pos = INSTR(line_text, "=")
            IF eq_pos > 0 THEN
                key_text = UCASE$(TRIM$(LEFT$(line_text, eq_pos - 1)))
                value_text = TRIM$(MID$(line_text, eq_pos + 1))
                IF key_text = wanted_key THEN
                    CLOSE #file_num
                    RETURN value_text
                END IF
            END IF
        END IF
    WEND

    CLOSE #file_num
    RETURN default_value

assist_ini_open_error:
    ON ERROR GOTO 0
    RETURN default_value
END FUNCTION

FUNCTION AssistLoadPack(pack_id AS STRING, pack_index AS INTEGER) AS INTEGER
    DIM base_path AS STRING
    DIM ini_path AS STRING
    DIM value_text AS STRING

    IF pack_index < 0 OR pack_index >= ASSIST_MAX_PACKS THEN RETURN 0
    pack_id = UCASE$(TRIM$(pack_id))
    IF pack_id = "" THEN RETURN 0

    base_path = AssistPathJoin(ASSIST_PACK_ROOT, pack_id)
    ini_path = AssistPathJoin(base_path, "PACK.INI")
    IF DIR(ini_path) = "" THEN RETURN 0

    g_assist_packs(pack_index).id = pack_id
    g_assist_packs(pack_index).title = AssistReadIniValue(ini_path, "TITLE", pack_id)
    g_assist_packs(pack_index).model_path = AssistReadIniValue(ini_path, "MODEL", "MODEL")
    g_assist_packs(pack_index).persona = AssistReadIniValue(ini_path, "PERSONA", "Helpful concise assistant.")
    value_text = AssistReadIniValue(ini_path, "HELP", "HELP.TXT")
    IF INSTR(value_text, "\") = 0 THEN value_text = AssistPathJoin(base_path, value_text)
    g_assist_packs(pack_index).help_path = value_text
    value_text = AssistReadIniValue(ini_path, "GOLDEN", "")
    IF value_text <> "" AND INSTR(value_text, "\") = 0 THEN value_text = AssistPathJoin(base_path, value_text)
    g_assist_packs(pack_index).golden_path = value_text
    value_text = AssistReadIniValue(ini_path, "USAGE", "USAGE.TXT")
    IF value_text <> "" AND INSTR(value_text, "\") = 0 THEN value_text = AssistPathJoin(base_path, value_text)
    g_assist_packs(pack_index).usage_path = value_text
    value_text = AssistReadIniValue(ini_path, "SPRITE", "")
    IF value_text <> "" AND INSTR(value_text, "\") = 0 THEN value_text = AssistPathJoin(base_path, value_text)
    g_assist_packs(pack_index).sprite_path = value_text
    value_text = AssistReadIniValue(ini_path, "ICONS", "")
    IF value_text <> "" AND INSTR(value_text, "\") = 0 THEN value_text = AssistPathJoin(base_path, value_text)
    g_assist_packs(pack_index).icons_path = value_text
    g_assist_packs(pack_index).actions = AssistReadIniValue(ini_path, "ACTIONS", "explain,more,cancel")
    g_assist_packs(pack_index).loaded = 1
    RETURN 1
END FUNCTION

SUB AssistUseBuiltinPack()
    g_assist_pack_count = 1
    g_assist_active_pack = 0
    g_assist_packs(0).id = "DEFAULT"
    g_assist_packs(0).title = "GPT2-BASIC Assistant"
    g_assist_packs(0).model_path = "MODEL"
    g_assist_packs(0).persona = "Helpful concise DOS assistant."
    g_assist_packs(0).help_path = ""
    g_assist_packs(0).golden_path = ""
    g_assist_packs(0).usage_path = ""
    g_assist_packs(0).sprite_path = ""
    g_assist_packs(0).icons_path = ""
    g_assist_packs(0).actions = "explain,chat,cancel"
    g_assist_packs(0).loaded = 1
END SUB

SUB AssistLoadPackList()
    DIM file_num AS INTEGER
    DIM line_text AS STRING

    g_assist_pack_count = 0
    g_assist_active_pack = 0

    IF DIR(ASSIST_PACK_LIST) = "" THEN
        AssistUseBuiltinPack
        RETURN
    END IF

    file_num = FREEFILE
    ON ERROR GOTO assist_pack_list_error
    OPEN ASSIST_PACK_LIST FOR INPUT AS #file_num
    ON ERROR GOTO 0

    WHILE EOF(file_num) = 0 AND g_assist_pack_count < ASSIST_MAX_PACKS
        LINE INPUT #file_num, line_text
        line_text = TRIM$(line_text)
        IF line_text <> "" AND LEFT$(line_text, 1) <> "#" AND LEFT$(line_text, 1) <> ";" THEN
            IF AssistLoadPack(line_text, g_assist_pack_count) <> 0 THEN
                g_assist_pack_count = g_assist_pack_count + 1
            END IF
        END IF
    WEND

    CLOSE #file_num
    IF g_assist_pack_count = 0 THEN AssistUseBuiltinPack
    RETURN

assist_pack_list_error:
    ON ERROR GOTO 0
    AssistUseBuiltinPack
END SUB

SUB AssistPrintPackList()
    DIM i AS INTEGER
    PRINT "Available packs:"
    FOR i = 0 TO g_assist_pack_count - 1
        PRINT "  "; AssistTrimFixed(g_assist_packs(i).id); " - "; AssistTrimFixed(g_assist_packs(i).title)
    NEXT i
END SUB

FUNCTION AssistSelectPack(pack_id AS STRING) AS INTEGER
    DIM i AS INTEGER
    DIM wanted AS STRING

    wanted = UCASE$(TRIM$(pack_id))
    FOR i = 0 TO g_assist_pack_count - 1
        IF AssistTrimFixed(g_assist_packs(i).id) = wanted THEN
            g_assist_active_pack = i
            RETURN 1
        END IF
    NEXT i

    RETURN 0
END FUNCTION

FUNCTION AssistTokenizerModeName(mode_value AS INTEGER) AS STRING
    IF mode_value = TOKENIZER_MODE_BYTE THEN RETURN "byte"
    IF mode_value = TOKENIZER_MODE_BPE THEN RETURN "bpe"
    IF mode_value = TOKENIZER_MODE_LEXICON THEN RETURN "lexicon"
    RETURN "unknown"
END FUNCTION

FUNCTION AssistInitializeModel(pack_index AS INTEGER) AS INTEGER
    DIM model_path AS STRING
    DIM vocab_path AS STRING

    IF pack_index < 0 OR pack_index >= g_assist_pack_count THEN RETURN 0
    model_path = AssistTrimFixed(g_assist_packs(pack_index).model_path)
    IF model_path = "" THEN model_path = "MODEL"

    IF GPT2BasicIsLoaded() <> 0 AND g_assist_model_path = model_path THEN
        IF g_assist_emit_records <> 0 THEN
            PRINT "ASSIST_MODEL|pack=" + AssistTrimFixed(g_assist_packs(pack_index).id) + _
                  "|path=" + AssistSafeText(model_path) + _
                  "|profile=" + AssistSafeText(GPT2BasicProfileName()) + _
                  "|tokenizer=" + AssistTokenizerModeName(g_tokenizer.tokenizer_mode) + _
                  "|ctx=" + LTRIM$(STR$(GPT2BasicContextLength())) + _
                  "|vocab=" + LTRIM$(STR$(GPT2BasicVocabSize())) + _
                  "|reuse=1"
        END IF
        RETURN 1
    END IF

    IF GPT2BasicIsLoaded() <> 0 THEN GPT2BasicFreeModel
    g_assist_model_path = ""

    InitializeDefaultTokenizer
    vocab_path = AssistPathJoin(model_path, "VOCAB.BIN")
    IF DIR(vocab_path) <> "" THEN
        ON ERROR GOTO assist_vocab_error
        LoadDefaultVocabulary vocab_path
        ON ERROR GOTO 0
    END IF

    IF GPT2BasicLoadModel(model_path) = 0 THEN
        IF g_assist_emit_records <> 0 THEN
            PRINT "ASSIST_MODEL_FAILED|pack=" + AssistTrimFixed(g_assist_packs(pack_index).id) + _
                  "|path=" + AssistSafeText(model_path)
        ELSE
            PRINT "Model load failed for "; model_path
        END IF
        RETURN 0
    END IF

    IF g_tokenizer.vocab_size <> GPT2BasicVocabSize() THEN
        IF g_assist_emit_records <> 0 THEN
            PRINT "ASSIST_MODEL_FAILED|pack=" + AssistTrimFixed(g_assist_packs(pack_index).id) + _
                  "|reason=vocab_mismatch|tokenizer=" + LTRIM$(STR$(g_tokenizer.vocab_size)) + _
                  "|model=" + LTRIM$(STR$(GPT2BasicVocabSize()))
        ELSE
            PRINT "Model/tokenizer vocabulary mismatch."
        END IF
        GPT2BasicFreeModel
        RETURN 0
    END IF

    g_assist_model_path = model_path
    IF g_assist_emit_records <> 0 THEN
        PRINT "ASSIST_MODEL|pack=" + AssistTrimFixed(g_assist_packs(pack_index).id) + _
              "|path=" + AssistSafeText(model_path) + _
              "|profile=" + AssistSafeText(GPT2BasicProfileName()) + _
              "|tokenizer=" + AssistTokenizerModeName(g_tokenizer.tokenizer_mode) + _
              "|ctx=" + LTRIM$(STR$(GPT2BasicContextLength())) + _
              "|vocab=" + LTRIM$(STR$(GPT2BasicVocabSize()))
    ELSE
        PRINT "Loaded model: "; model_path; " ("; GPT2BasicProfileName(); ")"
    END IF
    RETURN 1

assist_vocab_error:
    ON ERROR GOTO 0
    IF g_assist_emit_records <> 0 THEN
        PRINT "ASSIST_MODEL_FAILED|pack=" + AssistTrimFixed(g_assist_packs(pack_index).id) + _
              "|reason=vocab_load|path=" + AssistSafeText(vocab_path)
    ELSE
        PRINT "Tokenizer vocabulary load failed: "; vocab_path
    END IF
    RETURN 0
END FUNCTION

SUB AssistPreloadActivePackModel()
    DIM model_path AS STRING
    DIM pack_id AS STRING

    IF g_assist_active_pack < 0 OR g_assist_active_pack >= g_assist_pack_count THEN RETURN
    model_path = AssistTrimFixed(g_assist_packs(g_assist_active_pack).model_path)
    IF model_path = "" THEN model_path = "MODEL"
    IF GPT2BasicIsLoaded() <> 0 AND g_assist_model_path = model_path THEN RETURN

    pack_id = AssistTrimFixed(g_assist_packs(g_assist_active_pack).id)
    PRINT "Loading "; pack_id; " model before prompt..."
    IF AssistInitializeModel(g_assist_active_pack) = 0 THEN
        PRINT "Model unavailable for "; pack_id; ". You can switch packs or use retrieval replies."
    END IF
    PRINT
END SUB

SUB AssistShutdownModel()
    IF GPT2BasicIsLoaded() <> 0 THEN GPT2BasicFreeModel
    g_assist_model_path = ""
END SUB

FUNCTION AssistClassifyIntent(query AS STRING) AS STRING
    DIM q AS STRING

    q = LCASE$(query)
    IF INSTR(q, "config.sys") > 0 OR INSTR(q, "autoexec") > 0 OR INSTR(q, "memory") > 0 THEN RETURN "dos_memory"
    IF INSTR(q, "batch") > 0 OR INSTR(q, ".bat") > 0 OR INSTR(q, "command") > 0 THEN RETURN "dos_batch"
    IF INSTR(q, "rewrite") > 0 OR INSTR(q, "professional") > 0 OR INSTR(q, "formal") > 0 THEN RETURN "office_rewrite"
    IF INSTR(q, "summar") > 0 OR INSTR(q, "shorten") > 0 THEN RETURN "office_summary"
    IF INSTR(q, "help") > 0 OR INSTR(q, "explain") > 0 THEN RETURN "explain"
    RETURN "general_chat"
END FUNCTION

FUNCTION AssistActionsForIntent(intent_name AS STRING) AS STRING
    IF intent_name = "dos_memory" THEN RETURN "show_config,explain_xms,more,cancel"
    IF intent_name = "dos_batch" THEN RETURN "write_batch,explain_command,more,cancel"
    IF intent_name = "office_rewrite" THEN RETURN "rewrite,shorten,formalize,cancel"
    IF intent_name = "office_summary" THEN RETURN "summarize,bullets,shorten,cancel"
    IF intent_name = "explain" THEN RETURN "explain,example,more,cancel"
    RETURN AssistTrimFixed(g_assist_packs(g_assist_active_pack).actions)
END FUNCTION

FUNCTION AssistRetrieve(pack_index AS INTEGER, query AS STRING) AS STRING
    DIM help_path AS STRING
    DIM file_num AS INTEGER
    DIM line_text AS STRING
    DIM key_text AS STRING
    DIM title_text AS STRING
    DIM body_text AS STRING
    DIM first_pipe AS INTEGER
    DIM second_pipe AS INTEGER
    DIM q AS STRING
    DIM best_text AS STRING
    DIM best_score AS INTEGER
    DIM row_score AS INTEGER

    IF pack_index < 0 OR pack_index >= g_assist_pack_count THEN RETURN ""
    help_path = AssistTrimFixed(g_assist_packs(pack_index).help_path)
    IF help_path = "" OR DIR(help_path) = "" THEN RETURN ""

    q = LCASE$(query)
    best_text = ""
    best_score = 0
    file_num = FREEFILE
    ON ERROR GOTO assist_retrieve_error
    OPEN help_path FOR INPUT AS #file_num
    ON ERROR GOTO 0

    WHILE EOF(file_num) = 0
        LINE INPUT #file_num, line_text
        line_text = TRIM$(line_text)
        IF line_text <> "" AND LEFT$(line_text, 1) <> "#" THEN
            first_pipe = INSTR(line_text, "|")
            IF first_pipe > 0 THEN
                second_pipe = INSTR(first_pipe + 1, line_text, "|")
                IF second_pipe > first_pipe THEN
                    key_text = LCASE$(TRIM$(LEFT$(line_text, first_pipe - 1)))
                    title_text = TRIM$(MID$(line_text, first_pipe + 1, second_pipe - first_pipe - 1))
                    body_text = TRIM$(MID$(line_text, second_pipe + 1))
                    row_score = 0
                    IF INSTR(q, key_text) > 0 THEN row_score = LEN(key_text)
                    IF INSTR(LCASE$(body_text), q) > 0 AND row_score < 1 THEN row_score = 1
                    IF row_score > best_score THEN
                        best_score = row_score
                        best_text = title_text + ": " + body_text
                    END IF
                END IF
            END IF
        END IF
    WEND

    CLOSE #file_num
    IF best_text <> "" THEN RETURN best_text
    RETURN ""

assist_retrieve_error:
    ON ERROR GOTO 0
    RETURN ""
END FUNCTION

FUNCTION AssistGoldenReply(pack_index AS INTEGER, query AS STRING) AS STRING
    DIM golden_path AS STRING
    DIM file_num AS INTEGER
    DIM line_text AS STRING
    DIM prompt_text AS STRING
    DIM answer_text AS STRING
    DIM tab_pos AS INTEGER
    DIM q AS STRING
    DIM best_text AS STRING
    DIM best_score AS INTEGER
    DIM row_score AS INTEGER

    IF pack_index < 0 OR pack_index >= g_assist_pack_count THEN RETURN ""
    golden_path = AssistTrimFixed(g_assist_packs(pack_index).golden_path)
    IF golden_path = "" OR DIR(golden_path) = "" THEN RETURN ""

    q = LCASE$(TRIM$(query))
    best_text = ""
    best_score = 0
    file_num = FREEFILE
    ON ERROR GOTO assist_golden_error
    OPEN golden_path FOR INPUT AS #file_num
    ON ERROR GOTO 0

    WHILE EOF(file_num) = 0
        LINE INPUT #file_num, line_text
        line_text = TRIM$(line_text)
        IF line_text <> "" AND LEFT$(line_text, 1) <> "#" THEN
            tab_pos = INSTR(line_text, CHR$(9))
            IF tab_pos > 1 THEN
                prompt_text = LCASE$(TRIM$(LEFT$(line_text, tab_pos - 1)))
                answer_text = TRIM$(MID$(line_text, tab_pos + 1))
                row_score = 0
                IF q = prompt_text THEN
                    CLOSE #file_num
                    RETURN answer_text
                END IF
                IF LEN(prompt_text) > 3 AND INSTR(q, prompt_text) > 0 THEN row_score = LEN(prompt_text)
                IF LEN(q) > 3 AND INSTR(prompt_text, q) > 0 THEN row_score = LEN(q)
                IF row_score > best_score THEN
                    best_score = row_score
                    best_text = answer_text
                END IF
            END IF
        END IF
    WEND

    CLOSE #file_num
    RETURN best_text

assist_golden_error:
    ON ERROR GOTO 0
    RETURN ""
END FUNCTION

FUNCTION AssistGenerate(prompt AS STRING, max_tokens AS INTEGER) AS STRING
    DIM input_tokens() AS INTEGER
    DIM generated_tokens() AS INTEGER
    DIM input_count AS INTEGER
    DIM generated_count AS INTEGER
    DIM context_tokens() AS INTEGER
    DIM context_len AS INTEGER
    DIM next_token AS INTEGER
    DIM i AS INTEGER
    DIM decoded_text AS STRING

    IF GPT2BasicIsLoaded() = 0 THEN RETURN ""
    Encode prompt, input_tokens(), input_count
    AssistPrepareGenerationPrompt input_tokens(), input_count, max_tokens

    REDIM context_tokens(0 TO input_count + max_tokens - 1)
    REDIM generated_tokens(0 TO max_tokens - 1)
    FOR i = 0 TO input_count - 1
        context_tokens(i) = input_tokens(i)
    NEXT i
    context_len = input_count
    generated_count = 0

    GPT2BasicBeginGeneration input_count
    FOR i = 0 TO max_tokens - 1
        next_token = GPT2BasicNextToken(context_tokens(), context_len, 0.0, ASSIST_DEFAULT_TOP_P, ASSIST_DEFAULT_TOP_K)
        context_tokens(context_len) = next_token
        context_len = context_len + 1
        IF next_token = 0 THEN EXIT FOR
        generated_tokens(generated_count) = next_token
        generated_count = generated_count + 1
        IF i >= ASSIST_SENTENCE_STOP_MIN_TOKENS THEN
            IF TinyGPTTokenEndsSentence(next_token) <> 0 THEN EXIT FOR
        END IF
    NEXT i

    IF generated_count = 0 THEN RETURN ""
    decoded_text = Decode(generated_tokens(), generated_count)
    RETURN TRIM$(AssistCleanGeneratedText(decoded_text))
END FUNCTION

FUNCTION AssistCleanGeneratedText(raw_text AS STRING) AS STRING
    DIM lower_text AS STRING
    DIM cut_pos AS INTEGER
    DIM marker_pos AS INTEGER

    lower_text = LCASE$(raw_text)
    cut_pos = LEN(raw_text) + 1

    marker_pos = INSTR(lower_text, " user:")
    IF marker_pos > 0 AND marker_pos < cut_pos THEN cut_pos = marker_pos
    marker_pos = INSTR(lower_text, " assistant:")
    IF marker_pos > 0 AND marker_pos < cut_pos THEN cut_pos = marker_pos
    marker_pos = INSTR(lower_text, " note:")
    IF marker_pos > 0 AND marker_pos < cut_pos THEN cut_pos = marker_pos
    marker_pos = INSTR(lower_text, " prompt:")
    IF marker_pos > 0 AND marker_pos < cut_pos THEN cut_pos = marker_pos
    marker_pos = INSTR(lower_text, " reply:")
    IF marker_pos > 0 AND marker_pos < cut_pos THEN cut_pos = marker_pos
    marker_pos = INSTR(lower_text, " q:")
    IF marker_pos > 0 AND marker_pos < cut_pos THEN cut_pos = marker_pos
    marker_pos = INSTR(lower_text, " a:")
    IF marker_pos > 0 AND marker_pos < cut_pos THEN cut_pos = marker_pos

    IF cut_pos <= LEN(raw_text) THEN RETURN RTRIM$(LEFT$(raw_text, cut_pos - 1))
    RETURN raw_text
END FUNCTION

FUNCTION AssistTextHasRepeatedChunk(value AS STRING) AS INTEGER
    DIM lower_text AS STRING
    DIM chunk_len AS INTEGER
    DIM idx AS INTEGER
    DIM max_pos AS INTEGER
    DIM chunk AS STRING

    lower_text = LCASE$(TRIM$(value))
    IF LEN(lower_text) < 12 THEN RETURN 0

    FOR chunk_len = 4 TO 20
        max_pos = LEN(lower_text) - (chunk_len * 3) + 1
        IF max_pos > 0 THEN
            FOR idx = 1 TO max_pos
                chunk = MID$(lower_text, idx, chunk_len)
                IF TRIM$(chunk) <> "" THEN
                    IF MID$(lower_text, idx + chunk_len, chunk_len) = chunk THEN
                        IF MID$(lower_text, idx + (chunk_len * 2), chunk_len) = chunk THEN RETURN 1
                    END IF
                END IF
            NEXT idx
        END IF
    NEXT chunk_len

    RETURN 0
END FUNCTION

FUNCTION AssistGeneratedLooksBad(generated AS STRING, query AS STRING) AS INTEGER
    DIM text AS STRING
    DIM lower_text AS STRING
    DIM lower_query AS STRING
    DIM i AS INTEGER
    DIM ch AS STRING
    DIM previous_ch AS STRING
    DIM run_count AS INTEGER
    DIM alpha_count AS INTEGER
    DIM comma_count AS INTEGER

    text = TRIM$(generated)
    IF text = "" THEN RETURN 1
    IF LEN(text) < 4 THEN RETURN 1
    IF LEN(text) > 360 THEN RETURN 1

    lower_text = LCASE$(text)
    lower_query = LCASE$(TRIM$(query))

    IF LEFT$(lower_text, 6) = "reply:" THEN RETURN 1
    IF INSTR(lower_text, " user:") > 0 THEN RETURN 1
    IF INSTR(lower_text, " assistant:") > 0 THEN RETURN 1
    IF INSTR(lower_text, " prompt:") > 0 THEN RETURN 1
    IF INSTR(lower_text, " note:") > 0 THEN RETURN 1
    IF INSTR(lower_text, " q:") > 0 THEN RETURN 1
    IF INSTR(lower_text, " a:") > 0 THEN RETURN 1
    IF INSTR(lower_text, "use two brief sentences") > 0 THEN RETURN 1
    IF INSTR(lower_text, "use to brief sentences") > 0 THEN RETURN 1
    IF INSTR(lower_text, "brief sentences") > 0 THEN RETURN 1
    IF INSTR(lower_text, "small friendly dos chat assistant") > 0 THEN RETURN 1
    IF INSTR(lower_text, "concise dos") > 0 THEN RETURN 1
    IF INSTR(lower_text, "you are ") > 0 THEN RETURN 1
    IF INSTR(lower_text, "the assistant") > 0 THEN RETURN 1
    IF AssistTextHasRepeatedChunk(text) <> 0 THEN RETURN 1

    previous_ch = ""
    run_count = 0
    alpha_count = 0
    comma_count = 0
    FOR i = 1 TO LEN(text)
        ch = MID$(text, i, 1)
        IF ch >= "A" AND ch <= "Z" THEN alpha_count = alpha_count + 1
        IF ch >= "a" AND ch <= "z" THEN alpha_count = alpha_count + 1
        IF ch = "," THEN comma_count = comma_count + 1
        IF ch = previous_ch THEN
            run_count = run_count + 1
            IF run_count >= 4 THEN RETURN 1
        ELSE
            previous_ch = ch
            run_count = 1
        END IF
    NEXT i

    IF alpha_count < 3 THEN RETURN 1
    IF comma_count >= 4 THEN RETURN 1
    ch = RIGHT$(RTRIM$(text), 1)
    IF ch <> "." AND ch <> "!" AND ch <> "?" THEN RETURN 1

    IF lower_query <> "" AND LEN(lower_query) > 8 THEN
        IF lower_text = lower_query THEN RETURN 1
    END IF

    IF INSTR(lower_query, "repeat") > 0 THEN
        IF INSTR(lower_text, "repeat") = 0 AND INSTR(lower_text, "short") = 0 AND INSTR(lower_text, "reset") = 0 THEN RETURN 1
    END IF
    IF INSTR(lower_query, "bug") > 0 OR INSTR(lower_query, "debug") > 0 OR INSTR(lower_query, "stuck") > 0 OR INSTR(lower_query, "fix") > 0 THEN
        IF INSTR(lower_text, "bug") = 0 AND INSTR(lower_text, "debug") = 0 AND INSTR(lower_text, "fix") = 0 AND INSTR(lower_text, "step") = 0 AND INSTR(lower_text, "error") = 0 AND INSTR(lower_text, "command") = 0 AND INSTR(lower_text, "test") = 0 THEN RETURN 1
    END IF
    IF INSTR(lower_query, "prompt") > 0 OR INSTR(lower_query, "answer") > 0 THEN
        IF INSTR(lower_text, "prompt") = 0 AND INSTR(lower_text, "answer") = 0 AND INSTR(lower_text, "question") = 0 THEN RETURN 1
    END IF
    IF INSTR(lower_query, "local inference") > 0 OR INSTR(lower_query, "model weights") > 0 THEN
        IF INSTR(lower_text, "local") = 0 AND INSTR(lower_text, "inference") = 0 AND INSTR(lower_text, "model") = 0 AND INSTR(lower_text, "weights") = 0 THEN RETURN 1
    END IF
    IF INSTR(lower_query, "old computer") > 0 OR INSTR(lower_query, "486") > 0 OR INSTR(lower_query, "dos") > 0 THEN
        IF INSTR(lower_text, "dos") = 0 AND INSTR(lower_text, "local") = 0 AND INSTR(lower_text, "486") = 0 AND INSTR(lower_text, "hardware") = 0 AND INSTR(lower_text, "computer") = 0 THEN RETURN 1
    END IF
    IF INSTR(lower_query, "release") > 0 OR INSTR(lower_query, "artifact") > 0 OR INSTR(lower_query, "tag") > 0 OR INSTR(lower_query, "status") > 0 THEN
        IF INSTR(lower_text, "release") = 0 AND INSTR(lower_text, "artifact") = 0 AND INSTR(lower_text, "tag") = 0 AND INSTR(lower_text, "test") = 0 AND INSTR(lower_text, "status") = 0 THEN RETURN 1
    END IF
    IF INSTR(lower_query, "dpmi") > 0 OR INSTR(lower_query, "protected mode") > 0 THEN
        IF INSTR(lower_text, "dpmi") = 0 AND INSTR(lower_text, "protected") = 0 AND INSTR(lower_text, "dos") = 0 AND INSTR(lower_text, "memory") = 0 THEN RETURN 1
    END IF

    RETURN 0
END FUNCTION

FUNCTION AssistFallbackReply(pack_index AS INTEGER, intent_name AS STRING, query AS STRING, retrieved AS STRING) AS STRING
    DIM pack_id AS STRING
    DIM q AS STRING

    IF TRIM$(retrieved) <> "" THEN RETURN retrieved
    IF pack_index >= 0 AND pack_index < g_assist_pack_count THEN
        pack_id = AssistTrimFixed(g_assist_packs(pack_index).id)
    ELSE
        pack_id = "DEFAULT"
    END IF
    q = LCASE$(TRIM$(query))

    IF pack_id = "DOSHELP" THEN
        IF intent_name = "dos_batch" THEN RETURN "Use short batch files with IF EXIST checks, clear messages, and 8.3 names."
        IF INSTR(q, "dpmi") > 0 OR INSTR(q, "protected mode") > 0 THEN RETURN "Protected-mode DOS programs need a DPMI host such as CWSDPMI.EXE beside the program."
        IF INSTR(q, "autoexec") > 0 THEN RETURN "Keep AUTOEXEC.BAT short, keep PATH simple, and load resident tools only when needed."
        IF INSTR(q, "config.sys") > 0 THEN RETURN "Load HIMEM.SYS, use DOS=HIGH,UMB, and keep FILES and BUFFERS modest."
        RETURN "Ask about CONFIG.SYS, AUTOEXEC.BAT, memory, batches, or where the model files live."
    END IF

    IF pack_id = "OFFICE" THEN
        IF intent_name = "office_summary" THEN RETURN "Paste the text and ask for a short summary with names, dates, and next actions kept."
        IF INSTR(q, "professional") > 0 OR INSTR(q, "polite") > 0 THEN RETURN "Use direct, polite wording, keep the concrete fact, and end with the next action."
        IF INSTR(q, "clearer") > 0 OR INSTR(q, "clarify") > 0 THEN RETURN "State what happened, why it matters, and the next action in one short paragraph."
        IF INSTR(q, "shorten") > 0 THEN RETURN "Keep the decision and next action, then remove repeated explanation."
        RETURN "Paste a short line and ask for rewrite, summary, formal tone, or shortening."
    END IF

    IF INSTR(q, "repeat") > 0 OR INSTR(q, "loop") > 0 THEN
        RETURN "I will reset to a shorter answer. Ask one specific question and I will avoid repeating phrases."
    END IF
    IF INSTR(q, "bug") > 0 OR INSTR(q, "debug") > 0 OR INSTR(q, "fix") > 0 OR INSTR(q, "stuck") > 0 THEN
        RETURN "Start with the failing command, the expected result, and the first error line. Then test one small fix."
    END IF
    IF INSTR(q, "local inference") > 0 OR INSTR(q, "model weights") > 0 THEN
        RETURN "Local inference means the DOS program reads model weights and produces the answer on this machine."
    END IF
    IF INSTR(q, "weird") > 0 OR INSTR(q, "bad") > 0 THEN
        RETURN "Ask a shorter question, switch packs if needed, and treat strange output as a signal to retry."
    END IF
    IF INSTR(q, "prompt") > 0 OR INSTR(q, "answer") > 0 THEN
        RETURN "A prompt is what you type. An answer is the model output after it reads that prompt."
    END IF
    IF INSTR(q, "old computer") > 0 OR INSTR(q, "486") > 0 THEN
        RETURN "It matters because a tiny local model can run on old DOS-style hardware without a network."
    END IF
    IF INSTR(q, "release") > 0 OR INSTR(q, "artifact") > 0 OR INSTR(q, "tag") > 0 OR INSTR(q, "status") > 0 THEN
        RETURN "Check the tag target, release assets, checksums, and test result before calling the release done."
    END IF
    IF INSTR(q, "sad") > 0 OR INSTR(q, "worried") > 0 OR INSTR(q, "lonely") > 0 THEN
        RETURN "I can listen briefly. Name the worry, then choose one small next step."
    END IF
    IF INSTR(q, "joke") > 0 THEN RETURN "DOS smiled because it found its prompt."
    IF INSTR(q, "story") > 0 THEN RETURN "A tiny model woke up inside DOS and answered one prompt at a time."
    IF INSTR(q, "plan") > 0 THEN RETURN "Pick one goal, list three steps, then start with the smallest step."
    IF INSTR(q, "help") > 0 THEN RETURN "Ask one clear question, or use /packs to switch to DOSHELP or OFFICE."

    RETURN "I am here in DOS. Ask one short question or switch packs with /pack NAME."
END FUNCTION

SUB AssistPrepareGenerationPrompt(input_tokens() AS INTEGER, BYREF input_count AS INTEGER, max_tokens AS INTEGER)
    DIM context_limit AS INTEGER
    DIM reserve_count AS INTEGER
    DIM keep_count AS INTEGER
    DIM start_idx AS INTEGER
    DIM i AS INTEGER
    DIM trimmed_tokens() AS INTEGER

    IF input_count > 0 THEN
        IF input_tokens(input_count - 1) = 0 THEN input_count = input_count - 1
    END IF

    context_limit = GPT2BasicContextLength()
    reserve_count = max_tokens
    IF reserve_count < 1 THEN reserve_count = 1
    IF reserve_count >= context_limit THEN reserve_count = context_limit - 1
    keep_count = context_limit - reserve_count
    IF keep_count < 1 THEN keep_count = 1

    IF input_count > keep_count THEN
        REDIM trimmed_tokens(0 TO keep_count - 1)
        start_idx = input_count - keep_count
        FOR i = 0 TO keep_count - 1
            trimmed_tokens(i) = input_tokens(start_idx + i)
        NEXT i
        REDIM input_tokens(0 TO keep_count - 1)
        FOR i = 0 TO keep_count - 1
            input_tokens(i) = trimmed_tokens(i)
        NEXT i
        input_count = keep_count
    END IF

    IF input_count < 1 THEN
        REDIM input_tokens(0 TO 0)
        input_tokens(0) = 0
        input_count = 1
    END IF
END SUB

SUB AssistPrefillPrompt(input_tokens() AS INTEGER, input_count AS INTEGER)
    DIM i AS INTEGER
    DIM prefill_count AS INTEGER

    prefill_count = input_count - 1
    IF prefill_count <= 0 THEN RETURN

    PRINT "Thinking: ";
    PRINT "prompt "; prefill_count; " tokens"
    FOR i = 0 TO prefill_count - 1
        IF (i MOD 8) = 0 THEN
            IF i > 0 THEN PRINT
            PRINT "  ctx"; i + 1; ": ";
        END IF
        PRINT AssistVisibleToken(input_tokens(i)); " ";
        IF GPT2BasicPrefillToken(input_tokens(i), i) = 0 THEN EXIT FOR
    NEXT i
    PRINT
END SUB

FUNCTION AssistStreamGenerate(prompt AS STRING, max_tokens AS INTEGER) AS STRING
    DIM input_tokens() AS INTEGER
    DIM generated_tokens() AS INTEGER
    DIM input_count AS INTEGER
    DIM generated_count AS INTEGER
    DIM context_tokens() AS INTEGER
    DIM context_len AS INTEGER
    DIM next_token AS INTEGER
    DIM i AS INTEGER
    DIM decoded_text AS STRING
    DIM cleaned_text AS STRING
    DIM previous_text AS STRING
    DIM piece AS STRING
    DIM generated_text AS STRING
    DIM progress_text AS STRING
    DIM erase_idx AS INTEGER

    IF GPT2BasicIsLoaded() = 0 THEN RETURN ""
    Encode prompt, input_tokens(), input_count
    AssistPrepareGenerationPrompt input_tokens(), input_count, max_tokens

    REDIM context_tokens(0 TO input_count + max_tokens - 1)
    REDIM generated_tokens(0 TO max_tokens - 1)
    FOR i = 0 TO input_count - 1
        context_tokens(i) = input_tokens(i)
    NEXT i
    context_len = input_count
    previous_text = ""
    generated_text = ""
    generated_count = 0

    GPT2BasicBeginGeneration input_count
    AssistPrefillPrompt input_tokens(), input_count
    PRINT "Thinking: sampling output tokens"
    PRINT "Answer: ";
    FOR i = 0 TO max_tokens - 1
        progress_text = "<t" + LTRIM$(STR$(i + 1)) + ">"
        PRINT progress_text;
        next_token = GPT2BasicNextToken(context_tokens(), context_len, 0.0, ASSIST_DEFAULT_TOP_P, ASSIST_DEFAULT_TOP_K)
        FOR erase_idx = 1 TO LEN(progress_text)
            PRINT CHR$(8); " "; CHR$(8);
        NEXT erase_idx
        context_tokens(context_len) = next_token
        context_len = context_len + 1
        IF next_token = 0 THEN EXIT FOR
        generated_tokens(generated_count) = next_token
        generated_count = generated_count + 1

        decoded_text = Decode(generated_tokens(), generated_count)
        cleaned_text = AssistCleanGeneratedText(decoded_text)
        piece = ""
        IF LEN(cleaned_text) > LEN(previous_text) THEN piece = MID$(cleaned_text, LEN(previous_text) + 1)
        IF piece <> "" THEN
            PRINT piece;
            generated_text = generated_text + piece
        END IF
        previous_text = cleaned_text

        IF LEN(cleaned_text) < LEN(decoded_text) THEN EXIT FOR

        IF i >= ASSIST_SENTENCE_STOP_MIN_TOKENS THEN
            IF TinyGPTTokenEndsSentence(next_token) <> 0 THEN EXIT FOR
        END IF
    NEXT i

    PRINT
    RETURN TRIM$(generated_text)
END FUNCTION

SUB AssistRenderFrame()
    CLS
    PRINT "+------------------------------------------------------------+"
    PRINT "| GPT2-BASIC Assistant Shell                                 |"
    PRINT "| Pack-driven text UI; VGA sprite/icon slots are pack assets. |"
    PRINT "+------------------------------------------------------------+"
    PRINT
END SUB

SUB AssistRenderPackStatus()
    DIM p AS AssistantPack
    p = g_assist_packs(g_assist_active_pack)
    PRINT "Pack : "; AssistTrimFixed(p.id); " - "; AssistTrimFixed(p.title)
    PRINT "Model: "; AssistTrimFixed(p.model_path)
    IF AssistTrimFixed(p.usage_path) <> "" THEN PRINT "Usage: /about"
    IF AssistTrimFixed(p.sprite_path) <> "" THEN PRINT "Sprite asset: "; AssistTrimFixed(p.sprite_path)
    IF AssistTrimFixed(p.icons_path) <> "" THEN PRINT "Icon asset  : "; AssistTrimFixed(p.icons_path)
    IF g_assist_emit_records <> 0 THEN
        PRINT "ASSIST_PACK|id=" + AssistTrimFixed(p.id) + _
              "|title=" + AssistSafeText(AssistTrimFixed(p.title)) + _
              "|model=" + AssistSafeText(AssistTrimFixed(p.model_path)) + _
              "|sprite=" + AssistSafeText(AssistTrimFixed(p.sprite_path)) + _
              "|icons=" + AssistSafeText(AssistTrimFixed(p.icons_path))
    END IF
    PRINT
END SUB

SUB AssistPrintPackUsage(pack_index AS INTEGER)
    DIM usage_path AS STRING
    DIM file_num AS INTEGER
    DIM line_text AS STRING

    IF pack_index < 0 OR pack_index >= g_assist_pack_count THEN RETURN
    usage_path = AssistTrimFixed(g_assist_packs(pack_index).usage_path)
    IF usage_path = "" OR DIR(usage_path) = "" THEN
        PRINT "No usage instructions for this pack."
        RETURN
    END IF

    PRINT "+------------------------------------------------------------+"
    PRINT "| Pack instructions                                          |"
    PRINT "+------------------------------------------------------------+"
    file_num = FREEFILE
    ON ERROR GOTO assist_usage_error
    OPEN usage_path FOR INPUT AS #file_num
    ON ERROR GOTO 0

    WHILE EOF(file_num) = 0
        LINE INPUT #file_num, line_text
        PRINT line_text
    WEND

    CLOSE #file_num
    PRINT
    RETURN

assist_usage_error:
    ON ERROR GOTO 0
    PRINT "Could not read pack instructions."
END SUB

SUB AssistRenderReply(query AS STRING, use_generation AS INTEGER)
    DIM pack_index AS INTEGER
    DIM intent_name AS STRING
    DIM actions AS STRING
    DIM retrieved AS STRING
    DIM retrieved_note AS STRING
    DIM golden AS STRING
    DIM memory_reply AS STRING
    DIM memory_context AS STRING
    DIM note_text AS STRING
    DIM prompt AS STRING
    DIM generated AS STRING
    DIM bubble AS STRING
    DIM model_path AS STRING
    DIM model_ready AS INTEGER
    DIM reply_source AS STRING

    pack_index = g_assist_active_pack
    intent_name = AssistClassifyIntent(query)
    actions = AssistActionsForIntent(intent_name)
    retrieved = AssistRetrieve(pack_index, query)
    retrieved_note = retrieved
    golden = AssistGoldenReply(pack_index, query)
    memory_reply = AssistMemoryReply(query)
    memory_context = AssistMemoryContext()
    note_text = retrieved_note
    model_path = AssistTrimFixed(g_assist_packs(pack_index).model_path)
    model_ready = 0
    generated = ""
    bubble = ""
    reply_source = "fallback"

    IF use_generation <> 0 THEN
        PRINT "+------------------------------------------------------------+"
        PRINT "| Assistant                                                  |"
        PRINT "+------------------------------------------------------------+"
        IF memory_reply = "" AND golden = "" AND retrieved_note = "" AND (GPT2BasicIsLoaded() = 0 OR g_assist_model_path <> model_path) THEN
            PRINT "Loading model..."
        END IF
    END IF

    IF use_generation <> 0 OR g_assist_emit_records <> 0 THEN
        model_ready = AssistInitializeModel(pack_index)
    END IF

    IF model_ready = 0 AND use_generation <> 0 AND memory_reply = "" AND golden = "" AND retrieved_note = "" THEN
        IF g_assist_emit_records <> 0 THEN
            PRINT "ASSIST_REPLY|pack=" + AssistTrimFixed(g_assist_packs(pack_index).id) + _
                  "|intent=" + intent_name + "|status=model_unavailable"
        END IF
        bubble = AssistFallbackReply(pack_index, intent_name, query, retrieved_note)
        reply_source = "fallback"
    END IF

    IF bubble = "" AND memory_reply <> "" THEN
        bubble = memory_reply
        reply_source = "memory"
    END IF

    IF bubble = "" AND golden <> "" THEN
        bubble = golden
        reply_source = "golden"
    END IF

    IF bubble = "" AND retrieved_note <> "" THEN
        bubble = retrieved_note
        reply_source = "retrieval"
    END IF

    IF bubble = "" AND use_generation <> 0 AND model_ready <> 0 THEN
        PRINT "Thinking: checking model answer"
        prompt = "User: " + query
        IF memory_context <> "" THEN prompt = memory_context + " " + prompt
        IF note_text <> "" THEN prompt = prompt + " Note: " + note_text
        prompt = prompt + " Assistant:"
        generated = AssistGenerate(prompt, ASSIST_MAX_REPLY_TOKENS)
        IF AssistGeneratedLooksBad(generated, query) = 0 THEN
            bubble = generated
            reply_source = "model"
        END IF
    END IF

    IF bubble = "" THEN
        bubble = AssistFallbackReply(pack_index, intent_name, query, retrieved_note)
        reply_source = "fallback"
    END IF

    IF LEN(bubble) > 360 THEN bubble = LEFT$(bubble, 360)

    IF g_assist_emit_records <> 0 THEN
        PRINT "ASSIST_REPLY|pack=" + AssistTrimFixed(g_assist_packs(pack_index).id) + _
              "|intent=" + intent_name + _
              "|ui=text" + _
              "|query=" + AssistSafeText(query) + _
              "|source=" + reply_source + _
              "|actions=" + AssistSafeText(actions) + _
              "|retrieval=" + AssistSafeText(retrieved_note) + _
              "|golden=" + AssistSafeText(golden) + _
              "|memory=" + AssistSafeText(memory_context) + _
              "|generated=" + AssistSafeText(generated) + _
              "|answer=" + AssistSafeText(bubble)
    END IF
    IF use_generation = 0 THEN
        PRINT "+------------------------------------------------------------+"
        PRINT "| Assistant                                                  |"
        PRINT "+------------------------------------------------------------+"
        PRINT bubble
        PRINT
    ELSE
        PRINT "Answer: "; bubble
    END IF
    PRINT "[ "; actions; " ]"
    PRINT
    AssistRememberTurn query, bubble
    AssistAddHistory "ASSISTANT: " + bubble
END SUB

SUB AssistScriptedDemo()
    AssistRenderFrame
    PRINT "ASSIST_BEGIN|suite=pack-shell|version=1"
    AssistPrintPackList
    PRINT

    AssistSelectPack "CHAT"
    AssistRenderPackStatus
    AssistPrintPackUsage g_assist_active_pack
    AssistRenderReply "Hello, what can you do?", 0

    AssistSelectPack "DOSHELP"
    AssistRenderPackStatus
    AssistPrintPackUsage g_assist_active_pack
    AssistRenderReply "How do I tune CONFIG.SYS memory for this assistant?", 0

    AssistSelectPack "OFFICE"
    AssistRenderPackStatus
    AssistPrintPackUsage g_assist_active_pack
    AssistRenderReply "Rewrite this memo in a professional tone.", 0

    PRINT "ASSIST_END|packs=" + LTRIM$(STR$(g_assist_pack_count))
    AssistShutdownModel
END SUB

SUB AssistGuardProbe()
    AssistRenderFrame
    PRINT "ASSIST_BEGIN|suite=guard-probe|version=1"
    AssistPrintPackList
    PRINT

    AssistSelectPack "CHAT"
    AssistRenderPackStatus
    AssistPreloadActivePackModel
    AssistRenderReply "what are you", 1
    AssistRenderReply "what is the meaning of life", 1
    AssistRenderReply "thanks", 1

    PRINT "ASSIST_END|suite=guard-probe|pack=" + AssistTrimFixed(g_assist_packs(g_assist_active_pack).id)
    AssistShutdownModel
END SUB

SUB AssistStressProbe()
    AssistRenderFrame
    PRINT "ASSIST_BEGIN|suite=stress-probe|version=1"
    AssistPrintPackList
    PRINT

    AssistSelectPack "CHAT"
    AssistRenderPackStatus
    AssistPreloadActivePackModel
    AssistRenderReply "why did my answer repeat itself", 1
    AssistRenderReply "tell me why this old computer model matters", 1
    AssistRenderReply "make a tiny plan for fixing a bug", 1
    AssistRenderReply "what is the difference between a prompt and an answer", 1
    AssistRenderReply "can you explain what local inference means", 1
    AssistRenderReply "i feel stuck debugging this", 1
    AssistRenderReply "what should i do if the answer sounds weird", 1
    AssistRenderReply "give me a status update about a delayed release", 1
    AssistRenderReply "can you browse the internet from dos", 1
    AssistRenderReply "can we talk about games", 1
    AssistRenderReply "i am tired", 1
    AssistRenderReply "i feel lonely", 1
    AssistRenderReply "my name is Tyr", 1
    AssistRenderReply "what is my name", 1
    AssistRenderReply "we are working on the DOSBox assistant", 1
    AssistRenderReply "what are we working on", 1
    AssistRenderReply "i prefer short answers", 1
    AssistRenderReply "how should you answer me", 1
    AssistRenderReply "what did i just ask", 1
    AssistRenderReply "what do you remember", 1

    AssistSelectPack "DOSHELP"
    AssistRenderPackStatus
    AssistPreloadActivePackModel
    AssistRenderReply "how do i keep conventional memory free", 1
    AssistRenderReply "my autoexec is too long what should i change", 1
    AssistRenderReply "how should i clean autoexec.bat", 1
    AssistRenderReply "write a batch command that checks for model files", 1
    AssistRenderReply "why does protected mode need a dpmi host", 1
    AssistRenderReply "what does config.sys do", 1

    AssistSelectPack "OFFICE"
    AssistRenderPackStatus
    AssistPreloadActivePackModel
    AssistRenderReply "make this sentence sound professional: the release broke", 1
    AssistRenderReply "summarize this: tests passed but the tag was stale", 1
    AssistRenderReply "summarize: tests passed but dosbox needed a helper file", 1
    AssistRenderReply "shorten: we need to verify the release before publishing", 1
    AssistRenderReply "write a polite status update about a delayed build", 1
    AssistRenderReply "make this clearer: the artifact uploaded but the tag was stale", 1

    PRINT "ASSIST_END|suite=stress-probe|packs=" + LTRIM$(STR$(g_assist_pack_count))
    AssistShutdownModel
END SUB

SUB AssistInteractive()
    DIM command_text AS STRING
    DIM query AS STRING

    AssistRenderFrame
    AssistPrintPackList
    PRINT
    AssistRenderPackStatus
    PRINT "Commands: /about, /pack NAME, /packs, /memory, /remember KEY=VALUE, /forget, /u, /d, /home, /end, /h, /clear, /quit"
    PRINT
    AssistPreloadActivePackModel

    DO
        PRINT "> ";
        LINE INPUT command_text
        command_text = TRIM$(command_text)
        IF command_text = "" THEN
            ' no-op
        ELSEIF LCASE$(command_text) = "/quit" THEN
            EXIT DO
        ELSEIF LCASE$(command_text) = "/packs" THEN
            AssistPrintPackList
        ELSEIF LCASE$(command_text) = "/about" THEN
            AssistPrintPackUsage g_assist_active_pack
        ELSEIF LCASE$(command_text) = "/memory" THEN
            AssistPrintMemory
        ELSEIF LCASE$(command_text) = "/forget" THEN
            AssistClearMemoryFacts
            PRINT "Memory cleared."
            PRINT
        ELSEIF LEFT$(LCASE$(command_text), 10) = "/remember " THEN
            PRINT AssistRememberCommand(MID$(command_text, 11))
            PRINT
        ELSEIF LCASE$(command_text) = "/history" OR LCASE$(command_text) = "/h" THEN
            AssistRenderHistory
        ELSEIF LCASE$(command_text) = "/up" OR LCASE$(command_text) = "/u" THEN
            AssistHistoryPage ASSIST_HISTORY_PAGE
        ELSEIF LCASE$(command_text) = "/down" OR LCASE$(command_text) = "/d" THEN
            AssistHistoryPage -ASSIST_HISTORY_PAGE
        ELSEIF LCASE$(command_text) = "/home" THEN
            AssistHistoryPage ASSIST_HISTORY_MAX
        ELSEIF LCASE$(command_text) = "/end" THEN
            g_assist_history_scroll = 0
            AssistRenderHistory
        ELSEIF LCASE$(command_text) = "/clear" THEN
            AssistClearHistory
            AssistRenderFrame
            AssistRenderPackStatus
        ELSEIF LEFT$(LCASE$(command_text), 6) = "/pack " THEN
            IF AssistSelectPack(MID$(command_text, 7)) <> 0 THEN
                AssistRenderFrame
                AssistRenderPackStatus
                AssistPreloadActivePackModel
            ELSE
                PRINT "Unknown pack."
            END IF
        ELSE
            query = command_text
            AssistAddHistory "YOU: " + query
            AssistRenderReply query, 1
        END IF
    LOOP

    AssistShutdownModel
END SUB

SUB AssistMain()
    DIM command_line AS STRING

    RANDOMIZE TIMER
    AssistLoadPackList
    command_line = LCASE$(TRIM$(COMMAND$))
    g_assist_emit_records = 0

    IF command_line = "--scripted" OR command_line = "--probe" THEN
        g_assist_emit_records = 1
        AssistScriptedDemo
        RETURN
    END IF

    IF command_line = "--guard-probe" THEN
        g_assist_emit_records = 1
        AssistGuardProbe
        RETURN
    END IF

    IF command_line = "--stress-probe" THEN
        g_assist_emit_records = 1
        AssistStressProbe
        RETURN
    END IF

    AssistInteractive
END SUB

AssistMain
