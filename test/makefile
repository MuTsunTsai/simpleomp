
MAKEFLAGS += -j12

CXX := em++

SRCF := specs
TEMP := obj
TARGET := build

SRC := $(wildcard $(SRCF)/*.cpp)
OBJ := $(patsubst $(SRCF)/%.cpp,$(TEMP)/%.o,$(SRC))
DEP := $(patsubst $(SRCF)/%.cpp,$(TEMP)/%.d,$(SRC))
LDFLAGS :=\
	-L../dist\
	-Llib\
	-lsimpleomp\
	-lgtest\
	-sENVIRONMENT=node\
	-sINITIAL_MEMORY=10MB

# https://emscripten.org/docs/tools_reference/emcc.html
# https://github.com/emscripten-core/emscripten/blob/main/src/settings.js

CPPFLAGS := -I$(SRCF) -Ilib -I../include -std=c++20
CXXFLAGS := -Oz -flto -fopenmp -pthread -sDISABLE_EXCEPTION_CATCHING=1

ifeq ($(OS),Windows_NT)
# Error is still possible in parallel running, so we add extra protection
	MK = -@if not exist "$(@D)" mkdir "$(@D)" 2> NUL
	RM = rmdir /s /q
else
	MK = @mkdir -p "$(@D)"
endif

JS := $(TARGET)/test.js

.PHONY: all
all: $(JS)

$(JS): $(OBJ) makefile ../dist/libsimpleomp.a
	$(MK)
	@echo Compiling [33m$@[0m
	@$(LINK.cc) $(USRFLAGS) $(OBJ) -o $@
	@echo [33mTest compile complete![0m

$(TEMP)/%.o: $(SRCF)/%.cpp
	$(MK)
	@echo Compiling [32m$(patsubst $(SRCF)/%,%,$<)[0m
	@$(COMPILE.cc) -MMD -c $< -o $@

# Ignoring old dependencies that were removed
%.h: ;
%.hpp: ;
%.d: ;

-include $(DEP)

.PHONY: clean
clean:
	@$(RM) "$(TEMP)"
