# simple-alpha-zero
A preliminary project to test and explore an implementation of the Alpha Zero AI.

**Make sure Tensorflow is intalled and updated**, see https://www.tensorflow.org/install for more details.

**Running the game**
  - run main.py
    - make sure the file of the test network named "test_10_50_25" (or any saved model) is found

**Training the network**
  - run the function "run_and_save_model()" in "./best_model/AlphaZeroPlayer.py"

**Data format**
  - A string of length 10
    - First 9 characters for each board position (0-8)
      - character types -> {"X", "O", "*"}
    - Last character to indicate whose turn it is to make a move

**_NOTE_:** The neural network needs this to be translated into a np.array of values {1, -1, 0} respectively 


