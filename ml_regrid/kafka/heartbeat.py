"""
Heartbeat is a short message send by a component to announce its liveness
status. The implementation is to share the same Heartbeater for the same
configuration.
"""
import datetime
import json
import os


class _Heartbeater(object):
    """
    This class is responsible for instantiating a heartbeater and
    send actual messages.
    To get an instance of this class, call the `get_heartbeater` helper
    function. Do NOT use it directly as the helper function provides
    caching for efficiency.
    """
    def __init__(self, producer, topic, version, event):
        self.producer = producer
        self.topic = topic
        self.version = version
        self.event = event
        self.host = os.uname()[1]

    def beat(self, component, status='NORMAL', remarks='Heartbeat'):
        utcnow = datetime.datetime.utcnow()
        envelope = {
            'version': self.version,
            'event': self.event,
            'payload': {
                'component': component,
                'host': self.host,
                'timestamp': utcnow.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'status': status,
                'remarks': remarks,
            },
        }

        self.producer.send(self.topic, json.dumps(envelope))


# The overall Heartbeater cache.
# Indexed by a tuple of producer, topic, version and event.
_HEARTBEATERS = {}


def get_heartbeater(producer,
                    topic='systemHealth', version=1, event='Heartbeat'):
    """
    Helper function to retrieve an existing or create a new Heartbeater
    depending on the giving configurations.

    :param producer: The Kafka Producer object
    :param topic: The topic to send the heartbeat
    :param version: Message version.
    :param event: Message event.
    :return: A Heartbeater object.
    """
    key = (producer, topic, version, event)
    heartbeater = _HEARTBEATERS.get(key, None)
    if heartbeater is None:
        heartbeater = _Heartbeater(producer, topic, version, event)
        _HEARTBEATERS[key] = heartbeater

    return heartbeater
                
