from api_helper import NorenApiPy
import logging

#enable dbug to see request and responses
logging.basicConfig(level=logging.DEBUG)

#start of our program
api = NorenApiPy()

#set token and user id
#paste the token generated using the login flow described 
# in LOGIN FLOW of https://pi.flattrade.in/docs
usersession='63ed108c037529ecdcab5fd089923e35c0437fbfa2caf5031a87ebaef500d444'
userid = 'FZ03508'

ret = api.set_session(userid= userid, password = '', usertoken= usersession)

ret = api.get_limits()
 
print(ret)

