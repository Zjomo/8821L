// NOTE THIS VERSION IS FOR THE PUSH INTO INTERFACE AND PULL-BACK experiments, not for general use!

#include <Arduino.h>
#include <mbed.h>
#include "Portenta_H7_TimerInterrupt.h"

// NOTE: You have to select the board and the port under Tools, not in the drop down menu above the editor
// when compiling using the Arduino IDE version 2+ otherwise the files won't compile.
#define MOTOR_BOARD_LED PI_15
#define INSTRUMENT_LED_PIN PI_14

#define OUT_BUFFER_SIZE 512
#define IN_BUFFER_SIZE 48
#define HW_TIMER_INTERVAL_US 64 //100 is the default, 64 close to what can be run at most stably with this code.
#define READS_PER_SEND 14
#define NBR_DATA_POINTS 34
#define NBR_DATA_POINTS_2 34*2
#define NBR_DATA_POINTS_3 34*3
#define NBR_DATA_POINTS_5 34*5
#define NBR_DATA_POINTS_7 34*7
#define NBR_DATA_POINTS_9 34*9

#define DATA_OFFSET 20
#define POINTS_PER_SEND 476 // = NBR_DATA_POINTS*READS_PER_SEND

#define CALIBRATION_SCALE 0.00001

Portenta_H7_Timer ITimer(TIM16);
using namespace mbed;

uint8_t inBuffer[IN_BUFFER_SIZE];
uint8_t outBuffer[OUT_BUFFER_SIZE];

// Motor parameters which the main script needs to know of

long motorTargetSpeeds[3] = {0, 0, 0};
long targetPositions[3] = {32768,32768,32768};
long max_speed = 20000;
long motorPositions[3] = {32768,32768,32768}; // int should be enough
double motorSpeeds[3]  = { 0.0f, 0.0f, 0.0f};
float dutyCycle[3] = { 0.0f, 0.0f, 0.0f };  // Set to zero later

double lastError[3] = {0, 0, 0}; // Last error for derivative calculation in PID position


// PSD ARRAYS
//int ADC_results[16];  // Raw values recieved from the ADC
long PSD_Force_A[3] = {0,0,0};
long PSD_Force_B[3] = {0,0,0};
bool force_limit_reached = false;
int current_force = 0;
uint16_t distance = 32768;

uint8_t PSD_Force_A_saved_uint[4] = {0,0,0,0};
uint8_t PSD_Force_B_saved_uint[4] = {0,0,0,0};

uint8_t PSD_Position_A_saved_uint[6] = {0,0,0,0,0,0};
uint8_t PSD_Position_B_saved_uint[6] = {0,0,0,0,0,0};

//PSD variables
float dist_x = 0; // Force PSD reading separation.
float dist_y = 0;
//float local_align_fac = 0;
uint16_t photodiode_A = 0;
uint16_t photodiode_B = 0;


// PIEZO variables
uint8_t DAC_AX[3] = {16, 128, 0}; 
uint8_t DAC_AY[3] = {18, 128, 0};
uint8_t DAC_BX[3] = {20, 128, 0};
uint8_t DAC_BY[3] = {22, 128, 0};

float Calibration_Factors[4] = {0.0179,0.0167, 0.0181, 0.0194} // Updated from calibration factors the 7th of february 2025 , previous values {0.02505,0.02565,0.02755,0.0287};

uint16_t feedback_stepsize = 1;
uint8_t feedback_axis = 1;
uint16_t feedback_limits[2] = {0, 62000};
uint16_t force_reading_limits[2] = {0,0};
bool instrument_led = true;

uint16_t counter = 0;      // counts the number of times we have sent data
uint16_t start_idx = 0;    //
uint16_t data_counter = 0; // counts the number of times we have sampled.
//int distance = 32768;
unsigned long sampleTime;  // Time in millis for last sample
int serial_counter = 0;

void calculate_forces_factors(uint16_t offset){
  // TODO this can be made more effective.
  PSD_Force_A[2] = ((outBuffer[offset+11] << 8) | outBuffer[offset+10]);
  PSD_Force_A[1] = ((outBuffer[offset+9] << 8) | outBuffer[offset+8]);
  PSD_Force_A[0] = ((outBuffer[offset+7] << 8) | outBuffer[offset+6]);

  PSD_Force_B[2] = ((outBuffer[offset+23] << 8) | outBuffer[offset+22]);
  PSD_Force_B[1] = ((outBuffer[offset+21] << 8) | outBuffer[offset+20]);
  PSD_Force_B[0] = ((outBuffer[offset+19] << 8) | outBuffer[offset+18]);

}

void sampleADC2uint8_t_Buffer(uint16_t offset){
    read_adc_2_uint8_t(0, outBuffer, offset+24, offset+26); // photodiodes
    read_adc_2_uint8_t(1, outBuffer, offset+10, offset+22); // Force sum
    read_adc_2_uint8_t(2, outBuffer, offset+8, offset+20);  // Force Y
    read_adc_2_uint8_t(3, outBuffer, offset+6, offset+18);  // Force X
    read_adc_2_uint8_t(4, outBuffer, offset+4, offset+16);  // Position sum
    read_adc_2_uint8_t(5, outBuffer, offset+2, offset+14);  // position Y
    read_adc_2_uint8_t(7, outBuffer, offset, offset+12);    // Position X
    
    // Here we try to avoid a collision between the read and the write to the saved forces.
    // Subtract X from A
    if (outBuffer[offset+6] < PSD_Force_A_saved_uint[0]){
      outBuffer[offset+7]--;
    }
    outBuffer[offset+6] -= PSD_Force_A_saved_uint[0];
    outBuffer[offset+7] -= PSD_Force_A_saved_uint[1];

    // Subtract Y from A
    if (outBuffer[offset+8] < PSD_Force_A_saved_uint[2]){
      outBuffer[offset+9]--;  
    }
    outBuffer[offset+8] -= PSD_Force_A_saved_uint[2];
    outBuffer[offset+9] -= PSD_Force_A_saved_uint[3];

    // Subtract X from B
    if (outBuffer[offset+18] < PSD_Force_B_saved_uint[0]){
      outBuffer[offset+19]--;
    }
    outBuffer[offset+18] -= PSD_Force_B_saved_uint[0];
    outBuffer[offset+19] -= PSD_Force_B_saved_uint[1];

    // Subtract Y from B
    if (outBuffer[offset+20] < PSD_Force_B_saved_uint[2]){
      outBuffer[offset+21]--;  
    }
    outBuffer[offset+20] -= PSD_Force_B_saved_uint[2];
    outBuffer[offset+21] -= PSD_Force_B_saved_uint[3];

    //////////////////// Same for position////////////////////////////////

    /////////////////////// A ////////////////////////////////////

    // TODO Changed here to see what happens
    if (outBuffer[offset]<PSD_Position_A_saved_uint[0]){
      outBuffer[offset+1]--;
    }
    outBuffer[offset] -= PSD_Position_A_saved_uint[0];
    outBuffer[offset+1] -= PSD_Position_A_saved_uint[1];

    // Subtract Y from A
    if (outBuffer[offset+2] < PSD_Position_A_saved_uint[2]){
      outBuffer[offset+3]--;  
    }
    outBuffer[offset+2] -= PSD_Position_A_saved_uint[2];
    outBuffer[offset+3] -= PSD_Position_A_saved_uint[3];


    //////////////////////B///////////////////////////////////////
    if (outBuffer[offset+12]<PSD_Position_B_saved_uint[0]){
      outBuffer[offset+13]--;
    }
    outBuffer[offset+12] -= PSD_Position_B_saved_uint[0]; 
    outBuffer[offset+13] -= PSD_Position_B_saved_uint[1];

    // Subtract Y from A
    if (outBuffer[offset+14] < PSD_Position_B_saved_uint[2]){
      outBuffer[offset+15]--;  
    }
    outBuffer[offset+14] -= PSD_Position_B_saved_uint[2];
    outBuffer[offset+15] -= PSD_Position_B_saved_uint[3];

}


void sample_motors_and_dac_to_buffer(uint16_t start_idx){
  outBuffer[0+start_idx] = motorPositions[0] & 0xFF;
  outBuffer[1+start_idx] = (motorPositions[0] >> 8) & 0xFF;
  outBuffer[2+start_idx] = motorPositions[1] & 0xFF;
  outBuffer[3+start_idx] = (motorPositions[1]  >> 8) & 0xFF;
  outBuffer[4+start_idx] = motorPositions[2] & 0xFF;
  outBuffer[5+start_idx] = (motorPositions[2]  >> 8) & 0xFF;
  
  // Message - This is used for debugging to have some space to send data to the main computer.
  outBuffer[6+start_idx] = (targetPositions[2])& 0xFF; //)targetPositions[0]& 0xFF; //PSD_Force_B_saved_uint[2];
  outBuffer[7+start_idx] = ((targetPositions[2])  >> 8) & 0xFF;//(targetPositions[0] >> 8) & 0xFF; //PSD_Force_B_saved_uint[3];
  
  //Adding the DAC outputs here to be able to monitor them closely. 
  outBuffer[8+start_idx] = DAC_AX[2];
  outBuffer[9+start_idx] = DAC_AX[1];

  outBuffer[10+start_idx] = DAC_AY[2];
  outBuffer[11+start_idx] = DAC_AY[1];

  outBuffer[12+start_idx] = DAC_BX[2];
  outBuffer[13+start_idx] = DAC_BX[1];

  outBuffer[14+start_idx] = DAC_BY[2];
  outBuffer[15+start_idx] = DAC_BY[1];

  // Temperature of A, raw value
  // TODO read it properly from time to time
  outBuffer[16+start_idx] = dutyCycle[0];//ADC_results[14] & 0xFF;
  outBuffer[17+start_idx] = 0;//(ADC_results[14]  >> 8) & 0xFF;

}

void sample_adc_to_buffer(uint16_t start_idx){
  sampleTime = micros();

  sampleADC2uint8_t_Buffer(start_idx);

  // Write LSB before MSB for each variable
  outBuffer[28+start_idx] = counter & 0xFF;
  outBuffer[29+start_idx] = (counter >> 8) & 0xFF;
  
  outBuffer[30+start_idx] = (sampleTime) & 0xFF;
  outBuffer[31+start_idx] = (sampleTime >> 8) & 0xFF;
  outBuffer[32+start_idx] = (sampleTime >> 16) & 0xFF;
  outBuffer[33+start_idx] = (sampleTime >> 24) & 0xFF;

}

void timer_sample(void){
  sampleTime = micros();
  if (data_counter>=POINTS_PER_SEND){
    // TODO wait instead of returning to avoid losing the data
    return ;
  }
  start_idx = DATA_OFFSET+data_counter;
  sampleADC2uint8_t_Buffer(start_idx);

  // Write LSB before MSB for each variable
  outBuffer[28+start_idx] = counter & 0xFF;
  outBuffer[29+start_idx] = (counter >> 8) & 0xFF;
  
  outBuffer[30+start_idx] = (sampleTime) & 0xFF;
  outBuffer[31+start_idx] = (sampleTime >> 8) & 0xFF;
  outBuffer[32+start_idx] = (sampleTime >> 16) & 0xFF;
  outBuffer[33+start_idx] = (sampleTime >> 24) & 0xFF;
  calculate_forces_factors(start_idx);
  data_counter += NBR_DATA_POINTS;
  // Sample motor positions
  if (data_counter == NBR_DATA_POINTS){
    sample_motors_and_dac_to_buffer(2);
  }

}

void setup() {
  analogReadResolution(16);
  analogWriteResolution(16);
  setHigh();

  pinMode(INSTRUMENT_LED_PIN, OUTPUT);
  pinMode(MOTOR_BOARD_LED, OUTPUT);
  digitalWrite(INSTRUMENT_LED_PIN, HIGH);
  digitalWrite(MOTOR_BOARD_LED, HIGH);
  
  Serial.begin(5000000); // Serial communications to host computer.
  Serial.setTimeout(1);

  setup_ADC();
  setup_DAC();    
  
  // Set buffer values to 0.
  for (int i = 0;i<IN_BUFFER_SIZE;i++){
    inBuffer[i] = 0;
  }
  for (int i = 0;i<OUT_BUFFER_SIZE;i++){
    outBuffer[i] = 0;
  }
  outBuffer[0] = 123;
  outBuffer[1] = 123;

  initiateInterrupts();
  ITimer.attachInterruptInterval(HW_TIMER_INTERVAL_US, timer_sample);
  startAllPWM();

}

void loop() {
  // TODO try to figure out a way to ensure that the same index is not treated twice before sending data. The delay should fix most of this but maybe not all
  // trigger twice per sampling cycle

  if (data_counter == POINTS_PER_SEND){
    // This is too slow to run consitently at 20kHz(delay 50 us), does however work at a delay of 64 us.
    Serial.write(outBuffer, OUT_BUFFER_SIZE);
    data_counter = 0;
    return;
  }
  
  if (data_counter == NBR_DATA_POINTS_7){
      readInputBuffer();
      delayMicroseconds(10);
      return;
  }
  
  /*
  if (inBuffer[35]==1 && data_counter == NBR_DATA_POINTS_9 ){
      // Moved this to allow for slower movements
      // todo have the trigger frequency be dependent on the step_size and just move a little each time.
      constant_speed_move(DAC_BY, feedback_stepsize, feedback_limits);
    }
  */
  //if(data_counter == NBR_DATA_POINTS_9 && inBuffer[35] != 0){
  if (data_counter % (NBR_DATA_POINTS_9) == 0&& inBuffer[35] != 0){ // TODO This is an aggresive change to previous protocols, don't run it as standard.
    
    switch (inBuffer[35]) {
        // Basic constant speed moves
        case 1:
            constant_speed_move(DAC_AX, feedback_stepsize, feedback_limits);
            break;
        case 2:
            constant_speed_move(DAC_AY, feedback_stepsize, feedback_limits);
            break;
        case 3:
            constant_speed_move(DAC_BX, feedback_stepsize, feedback_limits);
            break;
        case 4:
            constant_speed_move(DAC_BY, feedback_stepsize, feedback_limits);
            break;

        // Move to force threshold, then stop
        // X axis - forward direction
        case 5:
            constant_speed_move_force_lims(DAC_AX, feedback_stepsize, force_reading_limits, 0);
            break;
        case 6:
            constant_speed_move_force_lims(DAC_BX, feedback_stepsize, force_reading_limits, 0);
            break;
        case 7:
            constant_speed_move_force_lims(DAC_AX, feedback_stepsize, force_reading_limits, 1);
            break;
        case 8:
            constant_speed_move_force_lims(DAC_BX, feedback_stepsize, force_reading_limits, 1);
            break;
        // Y axis - forward direction
        case 9:
            constant_speed_move_force_lims(DAC_AY, feedback_stepsize, force_reading_limits, 2);
            break;
        case 10:
            constant_speed_move_force_lims(DAC_BY, feedback_stepsize, force_reading_limits, 2);
            break;
        case 11:
            constant_speed_move_force_lims(DAC_AY, feedback_stepsize, force_reading_limits, 3);
            break;
        case 12:
            constant_speed_move_force_lims(DAC_BY, feedback_stepsize, force_reading_limits, 3);
            break;

        // Move to force threshold, then stop - reverse direction
        // X axis - reverse direction
        case 13:
            constant_speed_move_force_lims_reverse(DAC_AX, feedback_stepsize, force_reading_limits, 0);
            break;
        case 14:
            constant_speed_move_force_lims_reverse(DAC_BX, feedback_stepsize, force_reading_limits, 0);
            break;
        case 15:
            constant_speed_move_force_lims_reverse(DAC_AX, feedback_stepsize, force_reading_limits, 1);
            break;
        case 16:
            constant_speed_move_force_lims_reverse(DAC_BX, feedback_stepsize, force_reading_limits, 1);
            break;
        // Y axis - reverse direction
        case 17:
            constant_speed_move_force_lims_reverse(DAC_AY, feedback_stepsize, force_reading_limits, 2);
            break;
        case 18:
            constant_speed_move_force_lims_reverse(DAC_BY, feedback_stepsize, force_reading_limits, 2);
            break;
        case 19:
            constant_speed_move_force_lims_reverse(DAC_AY, feedback_stepsize, force_reading_limits, 3);
            break;
        case 20:
            constant_speed_move_force_lims_reverse(DAC_BY, feedback_stepsize, force_reading_limits, 3);
            break;

        // Default case to handle any unexpected values
        default:
            // Handle undefined case, if needed
            break;
    }


  }

  // If Piezos are to be updated, do so
  if (data_counter % (NBR_DATA_POINTS_2) == 0){
    if (inBuffer[25] == 1){
      // No point in doing this more often than what is needed
      // TODO calculate the necessary differences and use these to update the autoalign.
      // TODO turn off autialign as soon as the force exceeds the allowed limit.
      if (inBuffer[35]<5||inBuffer[35]>12){ // added an autoalign here which is supposed ensure that the second laser follows the first one in reversing the bead.
        // TODO Force limit is calculated a bit everywhere, do it only in one place.
        if (inBuffer[35]<16){
          current_force = int(PSD_Force_A[0]) - int(PSD_Force_B[0]); // + instead of -?
          force_limit_reached = abs(current_force) > force_reading_limits[0];
        }
        else{
          current_force = int(PSD_Force_A[1]-32768) + int(PSD_Force_B[1]-32768);
          force_limit_reached = abs(current_force) > force_reading_limits[0];
        }
        autoalign(0);
      }
      else if (inBuffer[35]<9){
        // check force along X
        current_force = int(PSD_Force_A[0]) - int(PSD_Force_B[0]); // + instead of -?
        force_limit_reached = abs(current_force) > force_reading_limits[0];
        if (!force_limit_reached){
          autoalign(0);
        }
      }
      else if (inBuffer[35]<13){
        // Check force along Y
        current_force = int(PSD_Force_A[1]-32768) + int(PSD_Force_B[1]-32768);
        force_limit_reached = abs(current_force) > force_reading_limits[0];
        if (!force_limit_reached){
          autoalign(0);
        }
      }
    }
    if (inBuffer[25] == 2){
      if (inBuffer[35]<5||inBuffer[35]>12){
        if (inBuffer[35]<16){
          current_force = int(PSD_Force_A[0]) - int(PSD_Force_B[0]); // + instead of -?
          force_limit_reached = abs(current_force) > force_reading_limits[0];
        }
        else{
          current_force = int(PSD_Force_A[1]-32768) + int(PSD_Force_B[1]-32768);
          force_limit_reached = abs(current_force) > force_reading_limits[0];
        }
        autoalign(1);
      }
      else if (inBuffer[35]<9){
        // Check force along X
        current_force = int(PSD_Force_A[0]) - int(PSD_Force_B[0]); // + instead of -?
        force_limit_reached = abs(current_force) > force_reading_limits[0];
        if (!force_limit_reached){
          autoalign(1);
        }
      }
      else if (inBuffer[35]<13){
        // Check force along Y
        current_force = int(PSD_Force_A[1]-32768) + int(PSD_Force_B[1]-32768);
        force_limit_reached = abs(current_force) > force_reading_limits[0];
        if (!force_limit_reached){
          autoalign(1);
        }
      }
    }
    writeNumberFast(DAC_AX);
    writeNumberFast(DAC_AY);
    writeNumberFast(DAC_BX);
    writeNumberFast(DAC_BY);
    return;
  }

  if (data_counter == NBR_DATA_POINTS_5){
      update_motor(2);
      update_motor(1);
  }
  if (data_counter == NBR_DATA_POINTS_3){
      update_motor(0);
      if (counter%256 == 100){// increased blinking time by factor 10, code too fast otherwise ;) 
      setLow();
      }
      else if (counter%256 == 0){    
        setHigh();
      }
      counter += 1;
  }
  // Adding a delay means that we get more consistent sampling.
  delayMicroseconds(16); // used to be 16
}



void readInputBuffer() {
  if (Serial.available() < IN_BUFFER_SIZE) {
    serial_counter += 1;
    if (serial_counter < 20){
      return;
    }
    // Try resetting the communications to the host
    Serial.end();
    delay(10);
    Serial.begin(5000000); // Serial communications to host computer.
    Serial.setTimeout(1);
    serial_counter = 0;
    
  }
  serial_counter = 0;
  Serial.readBytes(inBuffer, IN_BUFFER_SIZE);

  // Check control bytes to verify that we have the right sender
  if (inBuffer[0] != 123 || inBuffer[1] != 123) {
    return;
  }

  // Calculating target positions and speeds
  targetPositions[0] = ((inBuffer[2] << 8) | inBuffer[3]);
  targetPositions[1] = ((inBuffer[4] << 8) | inBuffer[5]);
  targetPositions[2] = ((inBuffer[6] << 8) | inBuffer[7]);
  motorTargetSpeeds[0] = ((inBuffer[8] << 8) | inBuffer[9]) - 32768; // Subtracting 32768 for conversion between uint and int.
  motorTargetSpeeds[1] = ((inBuffer[10] << 8) | inBuffer[11]) - 32768;
  motorTargetSpeeds[2] = ((inBuffer[12] << 8) | inBuffer[13]) - 32768;

  // TODO clean up the response protocol and use only one byte to decide which protocol combination to use.
  if (inBuffer[25] == 0&&inBuffer[35]==0){ 
    DAC_AX[1] = inBuffer[14];
    DAC_AX[2] = inBuffer[15];

    DAC_AY[1] = inBuffer[16];
    DAC_AY[2] = inBuffer[17];

    DAC_BX[1] = inBuffer[18];
    DAC_BX[2] = inBuffer[19];

    DAC_BY[1] = inBuffer[20];
    DAC_BY[2] = inBuffer[21];
  }
  else if (inBuffer[25] == 1 && inBuffer[35]==0){ 
      DAC_BX[1] = inBuffer[18];
      DAC_BX[2] = inBuffer[19];

      DAC_BY[1] = inBuffer[20];
      DAC_BY[2] = inBuffer[21];
    }
  
  else if (inBuffer[25] == 2 && inBuffer[35]==0){
    DAC_AX[1] = inBuffer[14];
    DAC_AX[2] = inBuffer[15];

    DAC_AY[1] = inBuffer[16];
    DAC_AY[2] = inBuffer[17];
  }
  // Check for other commands.
  read_feedback_method();

  if (inBuffer[34] == 0 && !instrument_led){
    // Check if the LED should be turned on or not.
    // PI_14
    digitalWrite(INSTRUMENT_LED_PIN, HIGH);
    instrument_led = true;
  }
  else if (instrument_led && inBuffer[34]){
    digitalWrite(INSTRUMENT_LED_PIN, LOW);
    instrument_led = false;
  }

  // Check if we are to update the zero levels of force or position.
  if (inBuffer[24] != 0){  
    set_parameters_from_buffer();
  }
  max_speed = ((inBuffer[44] << 8) | inBuffer[45]); // should probably not be any offset here

}
uint8_t calculate_force_average(uint16_t start_idx, uint8_t nbr_numbers){
  uint16_t sum = 0;
  for (uint8_t idx=0;idx<nbr_numbers;idx++){
    sum += outBuffer[start_idx-idx*NBR_DATA_POINTS];
  }
  //uint8_t result = sum / nbr_numbers;
  uint8_t result = (sum + (nbr_numbers / 2)) / nbr_numbers; // Adding half of nbr_numbers before division for rounding

  return result;
}

void set_parameters_from_buffer(){
  // This is really incosnsitent for some reaon... Don't know why
  if(inBuffer[24] == 1){
    // Spent a silly amount of time getting the += to work, decided to do it on the host computer instead.
    PSD_Force_A_saved_uint[0] = inBuffer[27]; 
    PSD_Force_A_saved_uint[1] = inBuffer[26] - 128;

    PSD_Force_A_saved_uint[2] = inBuffer[29];
    PSD_Force_A_saved_uint[3] = inBuffer[28]-128;

    PSD_Force_B_saved_uint[0] = inBuffer[31];
    PSD_Force_B_saved_uint[1] = inBuffer[30] - 128;

    PSD_Force_B_saved_uint[2] = inBuffer[33];
    PSD_Force_B_saved_uint[3] = inBuffer[32] - 128; 

  }

  else if(inBuffer[24] == 2){
    //Sets the PSDs to their original position.
    PSD_Force_B_saved_uint[0] = 0;
    PSD_Force_B_saved_uint[1] = 0;
    PSD_Force_B_saved_uint[2] = 0;
    PSD_Force_B_saved_uint[3] = 0;

    PSD_Force_A_saved_uint[0] = 0;
    PSD_Force_A_saved_uint[1] = 0;
    PSD_Force_A_saved_uint[2] = 0;
    PSD_Force_A_saved_uint[3] = 0;

  }

  else if(inBuffer[24] == 3){
    // put in the force calibration factors here
    Calibration_Factors[0] = float((inBuffer[26] << 8) | inBuffer[27]) * CALIBRATION_SCALE;
    Calibration_Factors[1] = float((inBuffer[28] << 8) | inBuffer[29]) * CALIBRATION_SCALE;
    Calibration_Factors[2] = float((inBuffer[30] << 8) | inBuffer[31]) * CALIBRATION_SCALE;
    Calibration_Factors[3] = float((inBuffer[32] << 8) | inBuffer[33]) * CALIBRATION_SCALE;
  }

  else if(inBuffer[24]==4){
    PSD_Position_A_saved_uint[0] = inBuffer[27]; 
    PSD_Position_A_saved_uint[1] = inBuffer[26] - 128;

    PSD_Position_A_saved_uint[2] = inBuffer[29];
    PSD_Position_A_saved_uint[3] = inBuffer[28]-128;

    PSD_Position_B_saved_uint[0] = inBuffer[31];
    PSD_Position_B_saved_uint[1] = inBuffer[30] - 128;

    PSD_Position_B_saved_uint[2] = inBuffer[33];
    PSD_Position_B_saved_uint[3] = inBuffer[32] - 128;
  }

  else if (inBuffer[24] == 5) {

    PSD_Position_A_saved_uint[0] = 0;
    PSD_Position_A_saved_uint[1] = 0;
    PSD_Position_A_saved_uint[2] = 0;
    PSD_Position_A_saved_uint[3] = 0;

    PSD_Position_B_saved_uint[0] = 0;
    PSD_Position_B_saved_uint[1] = 0;
    PSD_Position_B_saved_uint[2] = 0;
    PSD_Position_B_saved_uint[3] = 0;
    
  }
  inBuffer[24] = 0;
}

void read_feedback_method(){
  /*
  reads what type of feedback we are telling the motor to apply as well as the limits of it
  */
  if (inBuffer[35] == 0){
    return;
  }

  // Fixed position feedback
  if (inBuffer[35]<5){
    feedback_limits[0] = ((inBuffer[36] << 8) | inBuffer[37]);
    feedback_limits[1] = ((inBuffer[38] << 8) | inBuffer[39]);
    feedback_stepsize = ((inBuffer[40] << 8) | inBuffer[41]);
  }
  else{
    force_reading_limits[0] = ((inBuffer[36] << 8) | inBuffer[37]);// - 32768;
    force_reading_limits[1] = ((inBuffer[38] << 8) | inBuffer[39]);// - 32768;
    feedback_stepsize = ((inBuffer[40] << 8) | inBuffer[41]);
  }

}
