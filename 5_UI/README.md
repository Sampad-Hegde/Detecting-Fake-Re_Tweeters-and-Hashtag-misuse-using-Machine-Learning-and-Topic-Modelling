# 5. User Interface Server

### Backend :
1. Flask

### Frontend :
1. HTML
2. CSS
3. JavaScript

## Make these changes before Running the Server:
1. Change the appropriate Computer IP in these places
    1. Line numbers 128 and 134 in ```server.py``` file.
2. Paste the appropriate Twitter API tokens in ```Driver.py``` file.

## Running the Server

```shell
set Flask_APP=server
```

```shell
flask run -h 0.0.0.0
```
!!! Above command takes lot of time (nearly 30 minutes) so, be calm until the server is up and running 

## Now you can use the WEB interface for checking the user is Fake or not. after pressing the submit button it takes some time in order to fetch the tweet data and re-training the LDA