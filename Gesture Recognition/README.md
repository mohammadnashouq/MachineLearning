# Gesture-Recognition-


### How to use

- Clone or download the git repository.
    ```sh
    $ git clone https://github.com/darshitpurohit101/Gesture-Recognition-.git
    ```
- Create and activate a virtual environment:
    ```sh
    $ virtualenv venv
    $ source venv/bin/activate
    ```
- Install the requirements inside the app folder
    ```sh
    $ pip install -r requirements.txt
    ```
- The first execution will create automatically a log file where the we track our system.

### To run The whole App locally without internet access:

- go to offline_app folder and  run client_gui_for_local_runing.py file then you will see the GUI with text box where the captured lettered will be showed
and when you click on listen the app will convert the predected sentence to speech.

### to run the app locally using client-server architecture do the following:

- run the local_server.py file in the folder server_client_app\server on port 5000
- run the client_gui_for_local_runing.py file form the folder  server_client_app\client that will show you the GUI


### To run backend on the server do the following:

- upload server_client_app/server folder to the server and run the backend_server.py file.
