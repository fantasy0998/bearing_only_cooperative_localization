import numpy as np
from load_data import angle2dcm

# Test case 1: Check if the function returns a 3x3 numpy matrix
yaw = 0.1
pitch = 0.2
roll = 0.3
Rnav2body = angle2dcm(yaw, pitch, roll)
assert isinstance(Rnav2body, np.matrix)
assert Rnav2body.shape == (3, 3)

# Test case 2: Check if the function returns the correct transformation matrix
yaw = 0.1
pitch = 0.2
roll = 0.3
expected_result = np.matrix([[0.975170327201816, 0.097843395007256, -0.198669330795061],
     [-0.036957013524625, 0.956425085849232, 0.289629477625516],
     [0.218350663146334, -0.275095847318244, 0.936293363584199]])
Rnav2body = angle2dcm(yaw, pitch, roll, input_units='rad')
assert np.allclose(Rnav2body, expected_result)

# Test case 3: Check if the function handles input angles in degrees
yaw = 90
pitch = 15
roll = 0
expected_result = np.matrix([
   [0.000000000000000   ,0.965925826289068  ,-0.258819045102521],
  [-1.000000000000000   ,0.000000000000000   ,                0],
   [0.000000000000000   ,0.258819045102521   ,0.965925826289068]
])
Rnav2body = angle2dcm(yaw, pitch, roll, input_units='deg')
assert np.allclose(Rnav2body, expected_result)

# Test case 4: Check if the function handles a different rotation sequence
yaw = 0.1
pitch = 0.2
roll = 0.3
rotation_sequence = '123'
expected_result = np.nan
Rnav2body = angle2dcm(yaw, pitch, roll, rotation_sequence=rotation_sequence)
assert np.isnan(Rnav2body)

print("All tests passed!")

