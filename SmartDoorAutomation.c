#include <Servo.h>
Servo servo;

int rled = 2;
int gled = 3 ; 
int roominus = 6;
int roomoutus = 5;
int buzzer = 13;
int pb = 4;
int buttonstate = 0;
  
void setup(){
    Serial.begin(9600);
	pinMode(buzzer, OUTPUT);
    pinMode(rled, OUTPUT);
    pinMode(gled, OUTPUT);
    pinMode(pb,INPUT);
    servo.attach(9);
}  

void loop(){
  digitalWrite(rled, LOW);
  digitalWrite(gled, LOW);
  servo.write(0);
  
  long disroomout = distancefromultrasonic(roomoutus);
  long disroomin = distancefromultrasonic(roominus);
  Serial.print("disroomout: ");
  Serial.println(disroomout);
  Serial.print("disroomin: ");
  Serial.println(disroomin);
  
  if (disroomout < 50 && disroomin > 100){
    digitalWrite(gled, HIGH);
    for (int i = 0; i<=90; i=i+10)
    	servo.write(i);
    delay(500);
	}
  else if(disroomin < 50 && disroomout > 100){
    digitalWrite(gled, HIGH);
    for (int i = 0; i<=90; i=i+10)
    	servo.write(i); 
    delay(500);
  }
  else if(disroomin<100 && disroomout<50){
    digitalWrite(rled, HIGH);
    tone(buzzer,300,1000);
    buttonstate = digitalRead(pb);
    
    Serial.println("Buttonstate: "); 
    Serial.print(buttonstate);
    if (buttonstate == HIGH)
    {
	  noTone(buzzer);
      digitalWrite(gled, HIGH);
      for (int i = 0; i<=90; i=i+15)
    	servo.write(i);
      delay(1000);
    }
  } delay(100);
}   

long distancefromultrasonic(int pinno){

  pinMode(pinno, OUTPUT);
  digitalWrite(pinno,LOW);
  delay(0.002);
  digitalWrite(pinno,HIGH);
  delay(0.005);
  digitalWrite(pinno,LOW);
  
  pinMode(pinno, INPUT);
  
  long time = pulseIn(pinno,HIGH); 
  long distance = time*0.0344/2;
  return distance;

}
