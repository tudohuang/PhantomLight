#include <Arduino.h>
#include "MLD.h"
#include"matrix.h"


//IO    
#define LEDARRAY_D 2
#define LEDARRAY_C 3
#define LEDARRAY_B 4
#define LEDARRAY_A 5
#define LEDARRAY_G 6
#define LEDARRAY_DI 7
#define LEDARRAY_CLK 8
#define LEDARRAY_LAT 9
int relayPin1 = 12;
int relayPin2 = 10;
int tempo = 144;

int buzzer = 11;
int pinArray[] = {LEDARRAY_D, LEDARRAY_C, LEDARRAY_B, LEDARRAY_A, LEDARRAY_G, LEDARRAY_DI, LEDARRAY_CLK, LEDARRAY_LAT};

unsigned char Display_Buffer[2];
bool playMelody = false;
unsigned int Word1[32];



bool fire = false;  // 火的狀態
int notes = sizeof(melody) / sizeof(melody[0]) / 2;
bool lu = false;
int wholenote = (60000 * 4) / tempo;

int divide = 0, noteDuration = 0;
void SceneToWord(unsigned char Scene[16][16]){
  int i, k;
  unsigned int value;
  clearWord();
  for(i = 0; i < 16; i++){
    for(k = 0; k < 16; k++){
      if(i < 8){
        value = Scene[i][k] << (7 - i);
        Word1[15 - k] += value;
      } else {
        value = Scene[i][k] << (15 - i);
        Word1[31 - k] += value;
      }
    }
  }
}

void setup() {
  for (int i = 0; i < sizeof(pinArray)/sizeof(pinArray[0]); i++) {
    pinMode(pinArray[i], OUTPUT);
  }
  Serial.begin(9600);
  pinMode(buzzer, OUTPUT);
  pinMode(relayPin1, OUTPUT);
}

void Display(unsigned int dat[]) {
  unsigned char i;
  for(i = 0 ; i < 16 ; i++){
    digitalWrite(LEDARRAY_G, HIGH);
    Display_Buffer[0] = dat[i];
    Display_Buffer[1] = dat[i+16];
    Send(Display_Buffer[1]);
    Send(Display_Buffer[0]);
    digitalWrite(LEDARRAY_LAT, HIGH);					 
    delayMicroseconds(1);
    digitalWrite(LEDARRAY_LAT, LOW);
    delayMicroseconds(1);
    Scan_Line(i);							
    digitalWrite(LEDARRAY_G, LOW);
    delayMicroseconds(1000);		
  }	

}


void playTheMelody() {
  for (int thisNote = 0; thisNote < notes * 2; thisNote = thisNote + 2) {

    int divide = melody[thisNote + 1];
    int noteDuration;
    if (divide > 0) {
      noteDuration = (wholenote) / divide;
    } else if (divide < 0) {
      noteDuration = (wholenote) / abs(divide);
      noteDuration *= 1.5;  // dotted notes are 1.5 longer
    }

    tone(buzzer, melody[thisNote], noteDuration * 0.9);

    delay(noteDuration);

    noTone(buzzer);
  }
}
void Scan_Line(unsigned int m) {
  digitalWrite(LEDARRAY_D, m & 0x8 ? HIGH : LOW);
  digitalWrite(LEDARRAY_C, m & 0x4 ? HIGH : LOW);
  digitalWrite(LEDARRAY_B, m & 0x2 ? HIGH : LOW);
  digitalWrite(LEDARRAY_A, m & 0x1 ? HIGH : LOW);
}

void Send(unsigned int dat) {
  unsigned char i;
  digitalWrite(LEDARRAY_CLK, LOW);
  delayMicroseconds(1);	
  digitalWrite(LEDARRAY_LAT, LOW);
  delayMicroseconds(1);

  for(i = 0 ; i < 8 ; i++){
    digitalWrite(LEDARRAY_DI, (dat & 0x01) ? HIGH : LOW);
    delayMicroseconds(1);
    digitalWrite(LEDARRAY_CLK, HIGH);				  
    delayMicroseconds(1);
    digitalWrite(LEDARRAY_CLK, LOW);
    delayMicroseconds(1);		
    dat >>= 1;
  }

  // 增加延遲時間
  delayMicroseconds(10); // 調整此處的延遲時間
}




void clearWord(){
  for(int i = 0; i < 32; i++)
    Word1[i] = 0;
}



void fireEffect() {
  SceneToWord(Smile1);
  for (int kk = 0; kk < 10; kk++)
    Display(Word1);

  SceneToWord(Smile2);
  for (int kk = 0; kk < 10; kk++)
    Display(Word1);

  SceneToWord(Smile3);
  for (int kk = 0; kk < 10; kk++)
    Display(Word1);
  //delay(1000);  // 圖案之間的間隔時間（單位：毫秒），可以根據需要調整
}




void blankEffect() {
  SceneToWord(lblank);
  for (int kk = 0; kk < 100; kk++)
    Display(Word1);
  
 
  SceneToWord(blank);
  for (int kk = 0; kk < 100; kk++)
    Display(Word1);



}


void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    if (input == "incendio") {
      fire = true;
      playMelody = false;
    } else if (input == "aqua") {
      playMelody = true;
      fire = false;
    } else if (input == "arresto") {
      digitalWrite(relayPin1, HIGH);  // 打開繼電器，啟動馬達
      delay(4000);                    // 延遲4秒
      digitalWrite(relayPin1, LOW);   // 關閉繼電器，停止馬達
      playMelody = false;
      fire = false;
    }else if(input == "alohomora"){
      digitalWrite(relayPin2, HIGH);  // 打開繼電器，啟動馬達
      delay(4000);                    // 延遲4秒
      digitalWrite(relayPin2, LOW);   // 關閉繼電器，停止馬達
      playMelody = false;
      fire = false;
    }else if(input = "lumos"){
      lu = true;
      playMelody = false;
      fire = false;
    }
  }

  if (fire) {
    fireEffect();
    delay(1000);
    fireEffect();
    delay(5000);
    fire = false;
  } else if (playMelody) {
    playTheMelody();
    playMelody = false;  // Reset the flag after playing the melody
  }else if(lu){
    blankEffect();
    delay(100);
    lu = false;
  }
}


