DESTDIR = ../bin.$(UNAME)
INCLUDES = -I../include -I. -I/usr/include/opencv
LIBDIR = -L../lib.$(UNAME) -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu

include ../Makerules

VIEWSOURCES =	mesh_view.cc

OTHERSOURCES =	mesh_align.cc \
		mesh_cat.cc \
		mesh_cc.cc \
		mesh_check.cc \
		mesh_crunch.cc \
		mesh_filter.cc \
		mesh_hf.cc \
		mesh_info.cc \
		mesh_make.cc \
		mesh_shade.cc \
		grid_subsamp.cc \
		xf.cc

VIEWOFILES = $(addprefix $(OBJDIR)/,$(VIEWSOURCES:.cc=.o))
OTHEROFILES = $(addprefix $(OBJDIR)/,$(OTHERSOURCES:.cc=.o))
OFILES = $(VIEWOFILES) $(OTHEROFILES)

VIEWPROG = $(DESTDIR)/mesh_view$(EXE)
OTHERPROGS = $(addsuffix $(EXE), $(addprefix $(DESTDIR)/, $(OTHERSOURCES:.cc=)))
PROGS = $(OTHERPROGS) $(VIEWPROG)

default: $(PROGS)


LIBS += -ltrimesh -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml
$(PROGS) : -ltrimesh

$(VIEWPROG) : $(VIEWOFILES)
	$(LINK) $(GLLIBS)

$(OTHERPROGS) : $(DESTDIR)/%$(EXE) : $(OBJDIR)/%.o
	$(LINK)

clean :
	-rm -f $(OBJDIR)/*.o $(OBJDIR)/*.d
	-rmdir $(OBJDIR)

spotless : clean
	-rm -f $(PROGS)
	-rmdir $(DESTDIR)

