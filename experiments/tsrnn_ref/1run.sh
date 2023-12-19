#CUDA_VISIBLE_DEVICES=2 python run.py --train_data_path '/csl/shifeng/rcGAN/aep/AEP_hourly' --valid_data_path '/csl/shifeng/rcGAN/aep/AEP_hourly' --horizon 48 --max_iterations 10000
CUDA_VISIBLE_DEVICES=2 python run.py --train_data_path '/csl/shifeng/rcGAN/aep/DAYTON_hourly' --valid_data_path '/csl/shifeng/rcGAN/aep/DAYTON_hourly' --horizon 48 --max_iterations 10000
CUDA_VISIBLE_DEVICES=2 python run.py --train_data_path '/csl/shifeng/rcGAN/aep/DAYTON_hourly' --valid_data_path '/csl/shifeng/rcGAN/aep/DAYTON_hourly' --horizon 168 --max_iterations 10000
#CUDA_VISIBLE_DEVICES=2 python run.py --train_data_path '/csl/shifeng/rcGAN/aep/AEP_hourly' --valid_data_path '/csl/shifeng/rcGAN/aep/AEP_hourly' --horizon 168 --max_iterations 10000
