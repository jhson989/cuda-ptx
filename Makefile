CC = nvcc
PROGRAM = program.out
SRCS = main.cu
INCS = 

.PHONY : all run clean

all: ${PROGRAM}

${PROGRAM}: ${SRCS} ${INC} Makefile
	${CC} -o $@ ${SRCS}


run : ${PROGRAM}
	./${PROGRAM}

clean :
	rm ${PROGRAM}


