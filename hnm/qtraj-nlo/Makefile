#####################################################
# qTraj Makefile 
# Author(s): Michael Strickland
#####################################################

DIR_MAIN       = ./
DIR_SRC        = $(DIR_MAIN)build/src
DIR_H          = $(DIR_MAIN)build/include
DIR_BUILD      = $(DIR_MAIN)build
DIR_OBJ        = $(DIR_BUILD)/obj

DEBUG = 
OPTIMIZATION = -O2
FLOWTRACE =
LINK_OPTIONS = 
CFLAGS = $(DEBUG) $(OPTIMIZATION) $(FLOWTRACE) $(OPTIONS)
COMPILER = g++
# general case
LIBS_ARMA = -lfftw3_threads -lfftw3 -lgsl -lm -larmadillo
# following line is for frontera
#LIBS_ARMA = -Wl,-rpath,$(TACC_FFTW3_LIB) -L$(TACC_FFTW3_LIB) -L $(TACC_GSL_LIB) -L /home/local/lib64 -lfftw3_threads -lfftw3 -lgsl -lm -larmadillo
OPTIONS_ARMA = -std=c++11 
INCLUDES = -I $(DIR_H) 

LIBS = $(LIBS_ARMA)
OPTIONS = $(OPTIONS_ARMA)
C := $(shell find $(DIR_SRC) -name '*.cpp' -not -name '.*')
C_OBJ  = $(C:$(DIR_SRC)%.cpp=$(DIR_OBJ)%.o)
OBJ = $(C_OBJ)

EXE =\
qtraj

LIB =\
libqtraj.a

$(EXE): $(OBJ)
	@echo "Linking: $@ ($(COMPILER))"
	$(COMPILER) $(LINK_OPTIONS) -o $@ $^ $(LIBS) $(INCLUDES)

$(LIB): $(OBJ)
	@echo "Generating library: $(LIB) ($(COMPILER))"
	ar -rcs $(DIR_OBJ)/$(LIB) $(OBJ)

$(DIR_OBJ)%.o: $(DIR_SRC)%.cpp
	@[ -d $(DIR_OBJ) ] || mkdir -p $(DIR_OBJ)
	@echo "Compiling: $< ($(COMPILER))"
	$(COMPILER) $(CFLAGS) $(INCLUDES) -c -o $@ $<

tests: $(LIB)
	./scripts/runTests.sh	

clean:
	@echo "Object files, library, and executable deleted"
	if [ -d "$(DIR_OBJ)" ]; then rm -rf $(EXE) $(DIR_OBJ)/*; rmdir $(DIR_OBJ); fi

.SILENT:
