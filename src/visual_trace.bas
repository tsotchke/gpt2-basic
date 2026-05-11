' *******************************************************
' * GPT2-BASIC VGA Trace Visualizer                    *
' *******************************************************
' * Optional lab executable. Reads TRACE.LOG generated  *
' * by GPT2.EXE --trace, draws a Mode 13h token/progress*
' * view, and emits machine-readable VISUAL_* records.  *
' *******************************************************

DECLARE FUNCTION VisualTokenColor(token_id AS INTEGER) AS INTEGER
DECLARE FUNCTION VisualField(line_text AS STRING, field_name AS STRING) AS STRING
DECLARE FUNCTION VisualInt(text_value AS STRING, fallback_value AS INTEGER) AS INTEGER
DECLARE SUB VisualOpenLog(log_path AS STRING)
DECLARE SUB VisualEmit(record_text AS STRING)
DECLARE SUB VisualCloseLog()
DECLARE SUB VisualDrawTokenBar(position AS INTEGER, token_id AS INTEGER, row_base AS INTEGER)
DECLARE SUB VisualizeTrace(trace_path AS STRING, log_path AS STRING)

DIM SHARED g_visual_out_file AS INTEGER
DIM SHARED g_visual_out_open AS INTEGER

FUNCTION VisualTokenColor(token_id AS INTEGER) AS INTEGER
    DIM color_idx AS INTEGER

    color_idx = 16 + (ABS(token_id) MOD 216)
    IF color_idx < 1 THEN color_idx = 1
    IF color_idx > 255 THEN color_idx = 255
    RETURN color_idx
END FUNCTION

FUNCTION VisualField(line_text AS STRING, field_name AS STRING) AS STRING
    DIM key_text AS STRING
    DIM start_pos AS INTEGER
    DIM end_pos AS INTEGER

    key_text = field_name + "="
    start_pos = INSTR(line_text, key_text)
    IF start_pos <= 0 THEN RETURN ""
    start_pos = start_pos + LEN(key_text)
    end_pos = INSTR(start_pos, line_text, "|")
    IF end_pos <= 0 THEN
        RETURN MID$(line_text, start_pos)
    END IF
    RETURN MID$(line_text, start_pos, end_pos - start_pos)
END FUNCTION

FUNCTION VisualInt(text_value AS STRING, fallback_value AS INTEGER) AS INTEGER
    IF LEN(TRIM$(text_value)) = 0 THEN RETURN fallback_value
    RETURN VALINT(text_value)
END FUNCTION

SUB VisualOpenLog(log_path AS STRING)
    IF log_path = "" THEN RETURN
    g_visual_out_file = FREEFILE
    ON ERROR GOTO visual_log_open_error
    OPEN log_path FOR OUTPUT AS #g_visual_out_file
    g_visual_out_open = 1
    ON ERROR GOTO 0
    RETURN

visual_log_open_error:
    ON ERROR GOTO 0
    g_visual_out_open = 0
END SUB

SUB VisualEmit(record_text AS STRING)
    IF g_visual_out_open <> 0 THEN
        PRINT #g_visual_out_file, record_text
    ELSE
        PRINT record_text
    END IF
END SUB

SUB VisualCloseLog()
    IF g_visual_out_open <> 0 THEN
        CLOSE #g_visual_out_file
        g_visual_out_open = 0
    END IF
END SUB

SUB VisualDrawTokenBar(position AS INTEGER, token_id AS INTEGER, row_base AS INTEGER)
    DIM x AS INTEGER
    DIM y AS INTEGER
    DIM height AS INTEGER
    DIM color_idx AS INTEGER

    x = 8 + ((position MOD 152) * 2)
    y = row_base + ((position \ 152) * 12)
    IF y > 196 THEN y = 196
    height = 2 + (ABS(token_id) MOD 48)
    IF height > y - 2 THEN height = y - 2
    IF height < 1 THEN height = 1
    color_idx = VisualTokenColor(token_id)

    LINE (x, y)-(x + 1, y - height), color_idx, BF
END SUB

SUB VisualizeTrace(trace_path AS STRING, log_path AS STRING)
    DIM file_num AS INTEGER
    DIM line_text AS STRING
    DIM graphics_ok AS INTEGER
    DIM prompt_count AS INTEGER
    DIM generated_count AS INTEGER
    DIM token_id AS INTEGER
    DIM token_pos AS INTEGER
    DIM token_text AS STRING
    DIM field_text AS STRING
    DIM saw_trace AS INTEGER

    VisualOpenLog log_path
    VisualEmit "VISUAL_BEGIN|suite=gpt2-basic-vga-trace|version=1"
    VisualEmit "VISUAL_SOURCE|trace=" + trace_path

    file_num = FREEFILE
    ON ERROR GOTO trace_open_error
    OPEN trace_path FOR INPUT AS #file_num
    GOTO trace_opened

trace_open_error:
    ON ERROR GOTO 0
    VisualEmit "VISUAL_FAILED|stage=open_trace"
    VisualEmit "VISUAL_END|generated_tokens=0|prompt_tokens=0"
    VisualCloseLog
    RETURN

trace_opened:
    ON ERROR GOTO 0

    graphics_ok = 0
    ON ERROR GOTO visual_graphics_error
    SCREEN 13
    graphics_ok = 1
    GOTO visual_graphics_ready

visual_graphics_error:
    ON ERROR GOTO 0
    graphics_ok = 0
    VisualEmit "VISUAL_GRAPHICS|mode=13|status=unavailable"
    GOTO visual_graphics_after_setup

visual_graphics_ready:
    ON ERROR GOTO 0
    CLS
    LINE (0, 0)-(319, 199), 7, B
    LINE (4, 70)-(315, 70), 8
    LINE (4, 170)-(315, 170), 8
    VisualEmit "VISUAL_GRAPHICS|mode=13|status=ok|width=320|height=200|colors=256"

visual_graphics_after_setup:
    WHILE NOT EOF(file_num)
        LINE INPUT #file_num, line_text
        IF LEFT$(line_text, 11) = "TRACE_BEGIN" THEN
            saw_trace = 1
        ELSEIF LEFT$(line_text, 11) = "TRACE_MODEL" THEN
            VisualEmit "VISUAL_MODEL|" + MID$(line_text, 13)
        ELSEIF LEFT$(line_text, 11) = "TRACE_STAGE" THEN
            VisualEmit "VISUAL_STAGE|" + MID$(line_text, 13)
        ELSEIF LEFT$(line_text, 17) = "TRACE_INPUT_TOKEN" THEN
            field_text = VisualField(line_text, "id")
            token_id = VisualInt(field_text, 0)
            field_text = VisualField(line_text, "pos")
            token_pos = VisualInt(field_text, prompt_count)
            token_text = VisualField(line_text, "text")
            VisualEmit "VISUAL_TOKEN|kind=prompt|pos=" + LTRIM$(STR$(token_pos)) + _
                       "|id=" + LTRIM$(STR$(token_id)) + _
                       "|color=" + LTRIM$(STR$(VisualTokenColor(token_id))) + _
                       "|text=" + token_text
            IF graphics_ok <> 0 THEN VisualDrawTokenBar token_pos, token_id, 64
            prompt_count = prompt_count + 1
        ELSEIF LEFT$(line_text, 10) = "TRACE_STEP" THEN
            field_text = VisualField(line_text, "token")
            token_id = VisualInt(field_text, 0)
            field_text = VisualField(line_text, "step")
            token_pos = VisualInt(field_text, generated_count)
            token_text = VisualField(line_text, "text")
            VisualEmit "VISUAL_TOKEN|kind=generated|step=" + LTRIM$(STR$(token_pos)) + _
                       "|id=" + LTRIM$(STR$(token_id)) + _
                       "|color=" + LTRIM$(STR$(VisualTokenColor(token_id))) + _
                       "|text=" + token_text
            IF graphics_ok <> 0 THEN VisualDrawTokenBar token_pos, token_id, 164
            generated_count = generated_count + 1
        ELSEIF LEFT$(line_text, 9) = "TRACE_END" THEN
            VisualEmit "VISUAL_TRACE_END|" + MID$(line_text, 11)
        END IF
    WEND

    CLOSE #file_num

    IF graphics_ok <> 0 THEN
        SLEEP 1
        SCREEN 0
        WIDTH 80, 25
    END IF

    IF saw_trace = 0 THEN VisualEmit "VISUAL_WARN|reason=no_trace_begin"
    VisualEmit "VISUAL_END|generated_tokens=" + LTRIM$(STR$(generated_count)) + _
               "|prompt_tokens=" + LTRIM$(STR$(prompt_count))
    VisualCloseLog
END SUB

DIM trace_path AS STRING
DIM log_path AS STRING
DIM command_text AS STRING
DIM split_pos AS INTEGER

command_text = TRIM$(COMMAND$)
split_pos = INSTR(command_text, " ")
IF split_pos > 0 THEN
    trace_path = TRIM$(LEFT$(command_text, split_pos - 1))
    log_path = TRIM$(MID$(command_text, split_pos + 1))
ELSE
    trace_path = command_text
    log_path = "VISUAL.LOG"
END IF
IF trace_path = "" THEN trace_path = "TRACE.LOG"
IF log_path = "" THEN log_path = "VISUAL.LOG"
VisualizeTrace trace_path, log_path
