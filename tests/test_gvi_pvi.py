# TODO: run GVI, single client PVI and verify identical results


# run cli, no_plot, compare vs states

import subprocess

log = True
gvi_command = ". venv/bin/activate; python ../experiments/regression.py --no_plot -e 10 --name=gvi_integration_test"
pvi_command = ". venv/bin/activate; python ../experiments/pvi_regression.py --no_plot -e 10 --name=pvi_integration_test"

commands = [gvi_command, pvi_command]

for c in commands:
    process = subprocess.Popen(c, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    if output and error == None:
        print(output)
    elif error == None:
        print("SUCCESS: ran process successfully")
    else:
        print(error)
