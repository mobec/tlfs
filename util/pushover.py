#
#   Copyright 2018 Moritz Becher
#
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import os
import requests
import json
import socket

def notify(message: str, pushover_authentivation: str = "pushover.auth"):
    if not os.path.isfile(pushover_authentivation):
        return

    message = "[" + socket.gethostname() + "]: " + message

    url = "https://api.pushover.net/1/messages.json"
    user = ""
    token = ""

    with open(pushover_authentivation, 'r') as f:
        auth = json.load(f)
        user = auth["user"]
        token = auth["token"]

    requests.post(url, data={
        'user': user,
        'token': token,
        'message': message
    })