python detection/run.py -c detection/iTransformer2_AD.py --gpus 3

python detection/run.py -c detection/iRNN_AD.py --gpus 0

python detection/run.py -c detection/iMamba2_AD.py --gpus 2

python detection/run.py -c detection/NumerMoe_AD.py --gpus 3
python detection/run.py -c detection/NumerMoe_AD.py --gpus 2
python detection/run.py -c detection/NumerMoe_AD.py --gpus 1


python detection/run.py -c detection/LinearMOE_AD.py --gpus 1

python detection/run.py -c detection/FourierMOE_AD.py --gpus 1
