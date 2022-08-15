import http from 'k6/http';
import {
    check,
    group,
    sleep
} from 'k6';
export const options = {
    stages: [
            {duration: '30s', target: 10}, // simulate ramp-up of traffic from 1 to 10 users over 30 seconds. {
            {duration: '7m', target: 10}, // stay at 10 users for 7 minutes
            {duration: '3m', target: 0 }, // ramp-down to 0 users
    ],
    thresholds: {
        'http_req_duration': ['p(99)<2000'] // 99% of requests must complete below 2 s
    },
};
const fixed = ["I love you!", "I hate you!", "I am a Kubernetes Cluster!"]
var random_shuffler = [
    "I love you!",
    "I hate you!",
    "I am a Kubernetes Cluster!",
    "I ran to the store",
    "The students are very good in this class",
    "Working on Saturday morning is brutal",
    "How much wood could a wood chuck chuck if a wood chuck could chuck wood?",
    "A Wood chuck would chuck as much wood as a wood chuck could chuck if a woodchuck could chuck wood ",
    "Food is very tasty",
    "Welcome to the thunderdome"
];
const generator = (cacheRate) => {
    const rand = Math.random()
    const text = rand > cacheRate ?
        random_shuffler.map(value => ({
            value,
            sort: Math.random()
        }))
        .sort((a, b) => a.sort - b.sort)
        .map(({
            value
        }) => value) :
        fixed
    return {
        text
    }
}

const NAMESPACE = 'jordanmeyer'
const BASE_URL = `https://${NAMESPACE}.mids-w255.com`;
const CACHE_RATE = 0.95

export default () => {
    const healthRes = http.get(`${BASE_URL}/health`)
    check(healthRes, {
        'is 200': (r) => r.status === 200
    })
    const payload = JSON.stringify(generator(CACHE_RATE))
    const predictionRes = http.request('POST', `${BASE_URL}/predict`, payload)
    check(predictionRes, {
        'is 200': (r) => r.status === 200
    })
};