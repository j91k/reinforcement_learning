# Reinforcement Learning for Business Optimization

## Project Overview

This project demonstrates how reinforcement learning (RL) can optimize business decision-making through two simulations: the Taxi Game and the CartPole Game. These examples show how RL can improve efficiency, reduce costs, and manage risk in dynamic environments. By using real-world analogies, the project highlights the practical applications of RL in business scenarios.

## Objective

The primary goal of this project is to showcase the use of RL in optimizing processes and decision-making in businesses. The games chosen reflect common business problems:

- **Taxi Game**: Similar to logistics optimization, where RL learns to find the most efficient routes for pickups and drop-offs, mirroring real-world delivery route optimization.
- **CartPole Game**: Represents the challenge of maintaining balance in unstable markets, where RL can assist in managing risk under unpredictable conditions.

## Business Case

Businesses today face challenges such as optimizing operations, adapting to changing environments, and managing risks effectively. Reinforcement learning provides a solution by learning from real-time data and continuously improving decision-making processes. The project focuses on:

- **Efficiency**: RL optimizes tasks, reducing time and resources required for activities like route planning and inventory management.
- **Cost Reduction**: Through learned strategies, RL minimizes waste and reduces operational costs.
- **Adaptability**: RL adapts to evolving business conditions, helping businesses stay competitive in a dynamic market.
- **Risk Management**: RL enables businesses to make more stable decisions in uncertain environments, reducing potential risks.

## Benefits

By applying RL through simulations, this project will highlight key benefits for businesses:

- **Increased Efficiency**: RL algorithms optimize decision-making processes.
- **Cost Savings**: Operations become more cost-effective through waste reduction.
- **Enhanced Adaptability**: RL responds dynamically to changes in business conditions.
- **Improved Risk Management**: RL enables better risk mitigation in unpredictable environments.

## Project Structure

### 1. Taxi Game Simulation

The Taxi Game demonstrates how RL can optimize routes, similar to logistics and delivery in real business scenarios.

#### Implementation Details:
- Initially implemented using a Q-table approach due to the discrete state space of the Taxi-v3 environment.
- Q-learning was chosen for its simplicity, efficiency, and guaranteed convergence under certain conditions.
- Improvements include adjusted random action selection and implementation of a score threshold.

### 2. CartPole Game Simulation

The CartPole Game illustrates how RL can maintain balance in uncertain conditions, akin to risk management in volatile markets.

#### Implementation Details:
- Utilizes a Double DQN (Deep Q-Network) agent.
- Maintains two separate neural networks: an online network for action selection and a target network for action evaluation.
- The online network is updated regularly, while the target network is updated less frequently to improve stability.

### 3. Exploratory Data Analysis and Visualization

Includes visualizations and statistical analysis to understand the performance of the RL algorithms.

## Live Demo

Try out our Reinforcement Learning AI Comparison tool:
[Streamlit App]([https://j91k-reinforcement-learning-gaming-test-ifkvvl.streamlit.app/](https://j91k-reinforcement-learning-gaming-test-aaaemi.streamlit.app/))

## Features

- Interactive environment selection (Taxi and CartPole)
- Two gameplay modes: Random Key Press and Improved Model
- Real-time visualization of the game environment
- Action history display
- Performance metrics and visualizations:
  - Action Distribution Bar Chart
  - Pole Angle Over Time (for CartPole)
  - Cart Position Over Time (for CartPole)

## Technologies Used

- Python
- Streamlit
- Gymnasium (OpenAI Gym)
- TensorFlow
- NumPy
- Matplotlib
- Pillow (PIL)

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/j91k/reinforcement_learning.git

2. Install the required dependencies
   ```bash
   pip install -r requirements.txt

4. **Important:** Run the [improve_ml_algo.ipynb](https://github.com/j91k/reinforcement_learning/blob/main/improve_ml_algo.ipynb) & [taxi_ml.ipynb](https://github.com/j91k/reinforcement_learning/blob/main/taxi_ml.ipynb) in order to have your model saved.

5. **Important:** Run the [gaming_test.py](https://github.com/j91k/reinforcement_learning/blob/main/gaming_test.py) then use
   ```bash
   streamlit run gaming_test.py

You can also try our live demo: [Streamlit App]([https://j91k-reinforcement-learning-gaming-test-ifkvvl.streamlit.app/](https://j91k-reinforcement-learning-gaming-test-aaaemi.streamlit.app/))

6. Select the environment (Taxi or CartPole) and the gameplay mode (Random Key Press or Improved Model) in the Streamlit interface.

7. Click "Start Game" to begin the simulation and observe the RL agent's performance.
   
## Conclusion
This project demonstrates how reinforcement learning can transform business operations by optimizing decision-making, reducing operational costs, and enhancing adaptability and risk management. Through the Taxi Game and CartPole Game, we showcase the real-world applications of RL in logistics and risk management, providing actionable insights for businesses.

## Team Members
-Jimmy Kim
-Jinseo Baek
