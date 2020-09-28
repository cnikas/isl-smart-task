Code to get system output of [our approach]() for SMART Task.
Models can be downloaded [here](https://drive.google.com/drive/folders/1TvSJAQQswzmbUAtlCRNvBjgKjYoq344q?usp=sharing)

Usage: 
    
    `python smart_task_get_system_output.py resources_dir`

Where:
    
    `resources_dir` is the directory containing all required models and files.

Contents of resources dir: 

    `Bert Fine-Tuning category`: category classification model folder
    
    `Bert Fine-Tuning literal`: literal type classification model folder
    
    `Bert Fine-Tuning category`: resource type classification model folder
    
    `mapping.csv`: contains mapping of classes to integers
    
    `dbpedia_hierarchy.json`: dbpedia classes with level (depth) and children
    
    `test.json`: contains test questions
