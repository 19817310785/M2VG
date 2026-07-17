bash tools/dist_test.sh   configs/gres/M2VG-grefcoco.py 1 --load-from  work_dir/gres/M2VG-grefcoco/M2VG-grefcoco.pth 
bash tools/dist_test.sh  configs/refcoco/M2VG-B-refcoco/M2VG-B-refcoco.py  1 --load-from  work_dir/refcoco/M2VG-B-refcoco/M2VG-B-refcoco.pth
bash tools/dist_test.sh  configs/refcoco/M2VG-L-refcoco/M2VG-L-refcoco.py  1 --load-from  work_dir/refcoco/M2VG-L-refcoco/M2VG-L-refcoco.pth
bash tools/dist_test.sh  configs/refzom/M2VG-refzom.py  1 --load-from  work_dir/refzom/M2VG-refzom/M2VG-refzom.pth
bash tools/dist_test.sh  configs/rrefcoco/M2VG-rrefcoco.py  1 --load-from  work_dir/rrefcoco/M2VG-rrefcoco/M2VG-rrefcoco.pth