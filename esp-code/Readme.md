# ESP8266 Fire & Smoke Detection Alert System

## Overview

This project is an IoT-based Fire and Smoke Detection System built using an ESP8266 (NodeMCU). The system continuously monitors smoke levels using an MQ-series gas sensor connected to the analog pin and detects flames using a flame sensor connected to a digital pin.

When smoke or fire is detected:

* An alert is sent to a remote backend server using HTTPS.
* A buzzer can be activated remotely through HTTP APIs.
* The ESP8266 hosts a local web server that allows external applications to start or stop the buzzer.

---

## Features

### Smoke Detection

* Reads analog values from the MQ gas sensor.
* Detects smoke when the sensor value exceeds the configured threshold.
* Sends an alert to the backend server.

### Fire Detection

* Reads digital output from the flame sensor.
* Detects fire when the sensor output becomes LOW.
* Sends an alert to the backend server.

### Remote Buzzer Control

* Start buzzer remotely using HTTP API.
* Stop buzzer remotely using HTTP API.
* Buzzer generates periodic beep alerts.

### Local Web Server

Provides REST APIs:

| Method | Endpoint | Description            |
| ------ | -------- | ---------------------- |
| GET    | /        | Server Status          |
| GET    | /ip      | Get ESP8266 IP Address |
| POST   | /start   | Start Buzzer           |
| POST   | /pause   | Stop Buzzer            |

### Secure Backend Communication

* Uses HTTPS requests.
* Sends JSON alert data to backend.
* Supports cloud-hosted backend servers.

---

## Hardware Requirements

### Components

1. ESP8266 NodeMCU
2. MQ Smoke/Gas Sensor
3. Flame Sensor Module
4. Active Buzzer
5. Breadboard
6. Jumper Wires
7. 5V Power Supply

---

## Pin Connections

### Smoke Sensor

| MQ Sensor | ESP8266 |
| --------- | ------- |
| AO        | A0      |
| VCC       | 3.3V    |
| GND       | GND     |

### Flame Sensor

| Flame Sensor | ESP8266 |
| ------------ | ------- |
| DO           | D2      |
| VCC          | 3.3V    |
| GND          | GND     |

### Buzzer

| Buzzer   | ESP8266 |
| -------- | ------- |
| Positive | D5      |
| Negative | D6      |

---

## Network Configuration

Update the WiFi credentials in the code:

```cpp
const char* ssid = "HOTSPOT";
const char* password = "asdfghjkl";
```

---

## Backend Configuration

Update backend URL if needed:

```cpp
String backendUrl =
"https://your-backend-url/esp-sensor-trigger";
```

Alert payload format:

```json
{
  "deviceId": "ESP8266_01",
  "sensor": "fire",
  "status": "detected",
  "timestamp": 123456
}
```

---

## API Usage

### Check Device Status

```bash
curl http://ESP_IP/
```

Response:

```text
NodeMCU Buzzer Server Running
```

---

### Get Device IP

```bash
curl http://ESP_IP/ip
```

Response:

```json
{
  "ip": "192.168.43.168"
}
```

---

### Start Buzzer

```bash
curl -X POST http://ESP_IP/start
```

Response:

```json
{
  "success": true,
  "message": "Buzzer Started"
}
```

---

### Stop Buzzer

```bash
curl -X POST http://ESP_IP/pause
```

Response:

```json
{
  "success": true,
  "message": "Buzzer Stopped"
}
```

---

## System Workflow

1. ESP8266 connects to WiFi.
2. HTTP server starts.
3. Smoke sensor continuously monitors gas concentration.
4. Flame sensor continuously monitors fire.
5. If smoke is detected:

   * Alert sent to backend.
6. If fire is detected:

   * Alert sent to backend.
7. Backend may trigger:

   * Buzzer Start API
   * Buzzer Stop API
8. ESP8266 activates buzzer accordingly.

---

## Serial Monitor Output Examples

### Normal State

```text
ADC = 350 Voltage = 1.12
```

### Smoke Detected

```text
ADC = 950 Voltage = 3.06
🔥 SMOKE DETECTED!
HTTP Code: 200
```

### Fire Detected

```text
🔥 FIRE DETECTED!
HTTP Code: 200
```

### Buzzer Running

```text
BEEP ON
BEEP OFF
BEEP ON
BEEP OFF
```

---

## Libraries Used

```cpp
ESP8266WiFi
ESP8266WebServer
ESP8266HTTPClient
WiFiClientSecure
```

Install these libraries from Arduino IDE Library Manager before uploading the code.

---

## Future Improvements

* Telegram notifications
* Email alerts
* Firebase integration
* Multiple sensor support
* OLED display monitoring
* Mobile application dashboard
* Battery backup system
* MQTT support
* Real-time monitoring panel

---

## Author

Karan Gade

Final Year B.E. Electronics & Telecommunication

MERN Stack Developer | IoT Enthusiast

GitHub: github.com/KaranGade24
LinkedIn: linkedin.com/in/karan-gade
