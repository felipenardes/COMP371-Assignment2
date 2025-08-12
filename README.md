# COMP 371 – Assignment 1
**Name:** Felipe Nardes  
**Student ID:** 40243669 
**Project Title:** Maze Explorer  

## Description
This 3D interactive maze game expands on Assignment 1 by adding dynamic shadows, hierarchical animation, and more complex interactions.  
The player explores a textured maze in first-person view, lit by a dynamic flashlight that casts real-time shadows.  
A complex OBJ model (Heracles statue) is placed in the maze, and the finish point is blocked by a hierarchical door with a twisting knob.

## Controls
- `W` – Move forward  
- `S` – Move backward  
- `A` – Strafe left  
- `D` – Strafe right  
- `Mouse` – Look around (first-person camera)  
- `Shift` – Move faster  
- `E` – Interact with the door (open/close)  
- `[` – Decrease flashlight brightness  
- `]` – Increase flashlight brightness  
- `ESC` – Exit the application

## Features
- **Dynamic Lighting & Shadows**  
  - Phong lighting model with ambient, diffuse, and specular components  
  - Spotlight torch that follows the player and casts shadows using shadow mapping with 3×3 PCF filtering  
- **Hierarchical Animation**  
  - Door consists of a parent frame, rotating slab, and child knob that twists before opening  
- **Complex Model**  
  - Heracles statue loaded from OBJ file, scaled and oriented in the maze center  
- **Interactive Gameplay**  
  - Door toggles when player is near and presses `E`  
  - Adjustable torch brightness (`[` and `]`)  
  - Collision detection with maze walls, statue, and door  
- **Textured Environment**  
  - Distinct textures for walls, ground, door, and knob  
  - Textures not used in course tutorials  
- **Victory Condition**  
  - Reaching the finish point triggers a success message and exits the game

## Technical Stack
- **OpenGL** (Core profile 3.3)  
- **GLFW** (Window/context/input)  
- **GLEW** (Extension loader)  
- **GLM** (Math library)  
- **stb_image** (Texture loading)  
- **OBJ Loader** (Custom loader for `.obj` models)  

## How to Run
1. Build the project with your C++ compiler, linking the above libraries.  
2. Ensure the following directories are present in the project root:  
   - `Textures/` – contains all texture images used in the scene  
   - `Models/` – contains `heracles.obj` and any related model files  
3. Run the executable from the project root so relative texture/model paths resolve correctly.

## Assets
- Textures from [ambientCG](https://ambientcg.com/) (public domain)  
- Heracles OBJ model from [Sketchfab](https://sketchfab.com/) (public domain)  

## Screenshots
Screenshots of the scene can be found in the `screenshots/` directory:
- Door closed
- Door open with knob twisted
- Statue casting dynamic shadows
