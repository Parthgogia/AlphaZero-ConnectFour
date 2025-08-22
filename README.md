# AlphaZero Game AI

A two-player game-playing agent implementation inspired by DeepMind's AlphaZero. This system combines Monte-Carlo Tree Search (MCTS) with neural networks that learn through self-play, creating strong AI players that discover optimal strategies without human game knowledge.

## Overview

The agent repeatedly plays games against itself, using those self-play games to train a neural network that provides both policy guidance (which moves to consider) and value estimates (how good a position is). This learned knowledge then improves future MCTS searches, creating a virtuous cycle of improvement.

## Features

- **AlphaZero-inspired architecture**: MCTS coupled with neural network training via self-play
- **Two game implementations**: Tic-Tac-Toe and Connect Four with pre-trained models
- **Flexible training modes**:
  - **Sequential self-play**: Games played one after another (memory efficient)
  - **Parallel self-play**: Multiple games simulated simultaneously for faster training
- **Interactive gameplay**: Pygame interface for Connect Four
- **Training visualization**: Monitor learning progress and search behavior
- **Model checkpoints**: Pre-trained models ready for evaluation and play

## Games Included

### Tic-Tac-Toe
Classic 3x3 grid game - perfect for testing and demonstrating the algorithm on a simple domain.

### Connect Four
More complex 6x7 grid game with deeper strategy - showcases the system's ability to learn sophisticated gameplay.

## Training Modes

### Sequential Self-Play
- Games played one after another
- Lower memory footprint
- Suitable for limited GPU resources
- Simpler implementation and debugging

### Parallel Self-Play
- Multiple games simulated simultaneously
- Maximizes GPU utilization
- Significantly faster training
- Better for modern hardware setups

## Repository Structure

```
├── training/           # Training scripts and pipelines
├── models/            # Pre-trained model checkpoints
├── games/             # Game implementations and rules
├── mcts/              # Monte-Carlo Tree Search implementation
├── neural_net/        # Neural network architectures
├── evaluation/        # Model evaluation and testing utilities
├── visualization/     # Training monitoring and analysis tools
├── interface/         # Pygame Connect Four interface
└── utils/             # Helper functions and utilities
```

## Dependencies

- **PyTorch** (`torch`) - Neural network training and inference
- **NumPy** (`numpy`) - Numerical computations
- **Matplotlib** (`matplotlib`) - Training visualization and plotting
- **tqdm** - Progress bars for training loops
- **Pygame** - Interactive Connect Four interface
- **Standard library**: `sys`, `math`, `random`

## Quick Start

### Playing Against Trained Models

```bash
# Play Connect Four with Pygame interface
python interface/connect_four_gui.py

# Play Tic-Tac-Toe in terminal
python games/play_tictactoe.py
```

### Training New Models

```bash
# Train with sequential self-play (memory efficient)
python training/train_sequential.py --game connect_four --iterations 1000

# Train with parallel self-play (faster)
python training/train_parallel.py --game connect_four --parallel_games 8
```

### Evaluating Models

```bash
# Evaluate model strength
python evaluation/evaluate_model.py --model models/connect_four_final.pt

# Visualize training progress
python visualization/plot_training_stats.py --log training_logs/connect_four.log
```

## How It Works

1. **Self-Play Generation**: The agent plays games against itself using MCTS guided by the current neural network
2. **Training Data Collection**: Game positions, MCTS visit counts, and final outcomes are recorded
3. **Neural Network Training**: The network learns to predict both move probabilities and position values
4. **Iterative Improvement**: Updated network improves MCTS search, leading to stronger self-play games

## Model Architecture

The neural network takes game board positions as input and outputs:
- **Policy head**: Probability distribution over legal moves
- **Value head**: Expected game outcome from current position

This dual-headed architecture allows the network to both guide move selection and evaluate positions during MCTS search.

## Performance

Pre-trained models demonstrate strong gameplay:
- **Tic-Tac-Toe**: Plays optimally, never loses when going first
- **Connect Four**: Defeats random players >95% of the time, competitive against intermediate human players

## Contributing

Contributions welcome! Areas for improvement:
- Additional game implementations
- Training efficiency optimizations  
- Enhanced visualization tools
- Mobile/web interfaces

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Inspired by the groundbreaking work in:
- Silver, D. et al. "Mastering the game of Go with deep neural networks and tree search" (AlphaGo)
- Silver, D. et al. "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (AlphaZero)