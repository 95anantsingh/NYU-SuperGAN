import os
# os.system("conda config --append envs_dirs /scratch/as14229/envs_dirs/ ")
os.system('echo "SuperGAN_DATA=\'cd /home/as14229/Shared/SuperGAN/data/\'" >> out ')
os.system('echo "alias GAN=\'conda activate GAN\'" >> out')
# print("\nEnvironment Imported- Use 'GAN' to activate\n")
