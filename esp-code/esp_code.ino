#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>


#define FLAME_PIN D2


// curl -X POST http://192.168.43.168/start
// WiFi Credentials
const char* ssid = "HOTSPOT";
const char* password = "asdfghjkl";

// Buzzer Pins
const int positiveBuzzerPin = D5;
const int negativeBuzzerPin = D6;

// HTTP Server
ESP8266WebServer server(80);

// Buzzer State
bool altertStart1 = false;//smaoke
bool altertStart2 = false;//fire
bool buzzerEnabled = false;
bool buzzerEnabled2 = false;
bool buzzerState = false;
unsigned long lastToggle = 0;
const unsigned long beepInterval = 1000; // 1 second

// send sensor triger over net
#include <ESP8266HTTPClient.h>
#include <WiFiClientSecure.h>

String backendUrl = "https://5000-firebase-survillancesystem-1762004374655.cluster-cd3bsnf6r5bemwki2bxljme5as.cloudworkstations.dev/esp-sensor-trigger";

void sendSensorTrigger() {

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected");
    return;
  }

  WiFiClientSecure client;
  client.setInsecure();

  HTTPClient http;

  if (http.begin(client, backendUrl)) {

    http.addHeader("Content-Type", "application/json");

    String payload = String("{") +
      "\"deviceId\":\"ESP8266_01\"," +
      "\"sensor\":\"fire\"," +
      "\"status\":\"detected\"," +
      "\"timestamp\":" + String(millis()) +
      "}";

    int httpCode = http.POST(payload);

    if (httpCode > 0) {
      Serial.printf("HTTP Code: %d\n", httpCode);
      Serial.println(http.getString());
    } else {
      Serial.println(http.errorToString(httpCode));
    }

    http.end();
  }
}

// Start Buzzer Endpoint
void handleStart() {
  buzzerEnabled = true;
  buzzerEnabled2 = true;

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json",
              "{\"success\":true,\"message\":\"Buzzer Started\"}");

  Serial.println("Buzzer Started");
}

// Pause Buzzer Endpoint
void handlePause() {
  buzzerEnabled = false;
  buzzerState = false;

  digitalWrite(positiveBuzzerPin, LOW);
  digitalWrite(negativeBuzzerPin, LOW);

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json",
              "{\"success\":true,\"message\":\"Buzzer Stopped\"}");

  Serial.println("Buzzer Stopped");
}

// Root Endpoint
void handleRoot() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "text/plain", "NodeMCU Buzzer Server Running");
}

// 404 Endpoint
void handleNotFound() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(404, "application/json",
              "{\"success\":false,\"message\":\"Route Not Found\"}");
}

void handleGetIP() {

  String json = "{";
  json += "\"ip\":\"";
  json += WiFi.localIP().toString();
  json += "\"";
  json += "}";

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", json);
}

void setup() {
  Serial.begin(9600);

// Smoke Sensor setup
  pinMode(FLAME_PIN, INPUT);

// Buzzer setpu
  pinMode(positiveBuzzerPin, OUTPUT);
  pinMode(negativeBuzzerPin, OUTPUT);

  digitalWrite(positiveBuzzerPin, LOW);
  digitalWrite(negativeBuzzerPin, LOW);


  Serial.println();
  Serial.println("Connecting to WiFi...");

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi Connected");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // Routes
  server.on("/", HTTP_GET, handleRoot);
  server.on("/start", HTTP_POST, handleStart);
  server.on("/pause", HTTP_POST, handlePause);
  server.on("/ip", HTTP_GET, handleGetIP);

  server.onNotFound(handleNotFound);

  server.begin();

  Serial.println("HTTP Server Started");
}

void loop() {
  server.handleClient();

// MQ Sensor Logic (Analog)

int value = analogRead(A0);

float voltage = (value / 1023.0) * 3.3;

Serial.print("ADC = ");
Serial.print(value);

Serial.print("  Voltage = ");
Serial.println(voltage);

if (value > 900) {

  Serial.println("🔥 SMOKE DETECTED!");

  // buzzerEnabled = true;
  // sendSensorTrigger();  

  if(altertStart1 == false){
  altertStart1 == true;
  sendSensorTrigger();
  }

} 
// else {

//   Serial.println("✅ Normal");

//   buzzerEnabled = false;
// }


delay(500);

// SMOKE SENSOR LOGIC

int flame = digitalRead(D2);

if (flame == LOW) {

  Serial.println("🔥 FIRE DETECTED!");

  // buzzerEnabled = true;

  if(altertStart2 == false){
  altertStart2 == true;
  sendSensorTrigger();
  }


}

delay(2000);

// BUZZER ALTER LOGIC START AND PAUSE

  if (buzzerEnabled) {

  if( buzzerEnabled == false)return;
  
    unsigned long currentMillis = millis();

    if (currentMillis - lastToggle >= beepInterval) {
      lastToggle = currentMillis;

      buzzerState = !buzzerState;

      digitalWrite(positiveBuzzerPin, buzzerState ? HIGH : LOW);
      digitalWrite(negativeBuzzerPin, LOW);

      Serial.println(buzzerState ? "BEEP ON" : "BEEP OFF");
      buzzerEnabled2 = false;
    }
  } else {
    digitalWrite(positiveBuzzerPin, LOW);
    digitalWrite(negativeBuzzerPin, LOW);
  }
}