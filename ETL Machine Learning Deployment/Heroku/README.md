#### Car Price Prediction - Deploy on Heroku
**Notes:** 
1. For some reason, build fails at `scipy` while working with **Python-3.9**. Therefore, I had to specify **Python-3.6.13** separately using `runtime.txt`.
2. Do not forget `Procfile`.
3. After build is successful, if error `H14` occurs while accessing web-app, use command `heroku ps:scale web=1`.