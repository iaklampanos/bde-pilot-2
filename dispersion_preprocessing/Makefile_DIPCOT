PGFORT=/opt/pgi/linux86-64/2016/bin/pgfortran
COP=../cop
WRF_IN=
PREPROC_IN=
LOC_FILES=VANDALOS_FIL CERNAVODA_FIL KOZLODUY_FIL GROHNDE_FIL EMSLAND_FIL SIZEWELL_FIL HINKLEY-P_FIL ALMARAZ_FIL LOVIISA_FIL FORSMARK_FIL
PREPROCOUT_FILES=coordin DELTA.LOC input.dat luse.dat RELFIL SOLUTION.GRD OPTIONS
COORDS=./coords
METDIP=./metdip
TOPODIP=./topodip
FILM=./film
MK_DIR=$(shell pwd)
PRE_DIR=
FILM_DIR=
DIP_DIR=
DIP_MAKE=
DIP=./DIPCOT
NCMAKER=dipcot_out_netcdf

help::
	@echo "help"

init::
	@echo $(MK_DIR)
	

PREPROC_IN=
preproc_f:: init
	export LD_LIBRARY_PATH=/usr/local/lib:;\
	cd $(PRE_DIR);\
	$(COORDS) $(PREPROC_IN);\
	$(METDIP) $(PREPROC_IN);\
	$(TOPODIP) $(PREPROC_IN);\

filmaker_f:: init
	cd $(PRE_DIR); mv $(PREPROCOUT_FILES) $(FILM_DIR);\
	cd $(FILM_DIR);\
	$(FILM)


POST_DIR=../post
dipcot_f:: init
	cd $(FILM_DIR); mv *FIL $(DIP_DIR);\
	cd $(DIP_DIR);\
	for l in $(LOC_FILES); do ln -fs $$l SRCFIL;\
	$(DIP);\
	rm -f DIPCOT.OUT;\
	cd $(POST_DIR);\
	outdir=`echo $(PREPROC_IN) | awk -F"_" '{print $$3"_"$$4}'`;\
	mkdir $$outdir;\
	./$(NCMAKER) $$l.nc;\
	mv $$l.nc $$outdir/;\
	cd $(DIP_DIR);\
	done;\

# DIPCOT_DEF=ALMARAZ_FIL|DIPCOT|DPgamdata.DAT|FORSMARK_FIL|KOZLODUY_FIL|SIZEWELL_FIL|CERNAVODA_FIL|DISFIL|DRYFIL|GROHNDE_FIL|LOVIISA_FIL|YLDFIL|VANDALOS_FIL|DOSFIL|EMSLAND_FIL|HINKLEY_FIL|OPTIONS
clean:: clean_dipcot
clean_dipcot::
	rm -rf $(DIP_DIR) && cp -R `dirname $(DIP_DIR)`/run_vanilla $(DIP_DIR);

preproc:: init
	@echo --- $(MK_DIR)
	echo `pwd`;\
	cd $(filmaker_preprocess);\
	export LD_LIBRARY_PATH=/usr/local/lib:;\
	for f in `ls $(WRF_IN)`; do make -f $(MK_DIR)/Makefile preproc_f PREPROC_IN=$(WRF_IN)/$$f WRF_IN=$(WRF_IN); done
	cd $(MK_DIR)
