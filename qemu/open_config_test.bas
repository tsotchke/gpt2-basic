DIM f AS INTEGER
DIM linebuf AS STRING

PRINT "open config diagnostic"
PRINT "dir CONFIG.TXT=["; DIR$("CONFIG.TXT"); "]"

f = FREEFILE
ON ERROR GOTO open_error
OPEN "CONFIG.TXT" FOR INPUT AS #f
PRINT "OPEN_OK file="; f

WHILE NOT EOF(f)
    LINE INPUT #f, linebuf
    PRINT "LINE=["; linebuf; "]"
WEND

CLOSE #f
PRINT "READ_OK"
END 0

open_error:
PRINT "OPEN_FAILED err="; ERR
END 1
