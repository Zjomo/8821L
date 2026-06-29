#define LDAC	         PE_2 // Needs to be put low, used if all outputs are to be changed at the same time
#define IOVDD_DAC      PI_3 // external power 
#define SPI_SYNC       PJ_9 //UART3 RX PIN
#define AUTO_ALIGN_LIM 0  // 40 is standard, seem to work fine. Used 20 before


#define AX  0.0501
#define AY  0.0513
#define BX  0.0551
#define BY  0.0574

#define ABX = 0.91
#define ABY = 0.89
#define BAX = 1.10
#define BAY = 1.12


float CONSTANT_FORCE_FAC = -0.02;
float AUTOALIGN_FAC = 0.0002;

float local_align_fac = 0;
float dist_x = 0;
float dist_y = 0;
int force_difference = 0; // Used in the constant force protocol to estimate the difference between current and target force in piezo steps.
uint16_t DAC_pos = 32768; // Variable used to for the DAC position in uint16 instead of 2 x uint8 
long move_counter = 0;


SPI spi(PC_3, PC_2, PI_1);

void setup_DAC() {
  pinMode(SPI_SYNC, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  GPIOJ->MODER |= (1 << (9 * 2)); // set sync as output pin

  pinMode(IOVDD_DAC,OUTPUT);
  digitalWrite(IOVDD_DAC, HIGH);
  pinMode(LDAC, OUTPUT);
  digitalWrite(LDAC, LOW);
  spi.frequency(25000000);           // Set up your frequency. Max is 25 MHz according to datasheet when operating at a logic voltage level of 3.4V as the portenta does.
  spi.format(8, 3);                  // Messege length (bits), SPI_MODE - check these in your SPI decice's data sheet.
}

void writeNumberFast(uint8_t DACData[]){
  /*
  Writes a number to the DAC. 
  Does this rather fast, at 25 MHz we can write all the 4 channels in ca 10 microseconds.
   */
  GPIOJ->ODR &= ~(1 << 9); // set sync low
  spi.write((const char*)DACData, 3, NULL, 0);
  GPIOJ->ODR |= (1 << 9); // set sync high
}

void autoalign(uint8_t trap){
  /*
  Moves one trap to match the position of the other by reducing the difference in the force reading between the two
  */
  // Trap = 0 => move trap A to match trap B
  local_align_fac = AUTOALIGN_FAC*((PSD_Force_A[2]-32768) + (PSD_Force_B[2]-32768));
  dist_x = Calibration_Factors[0]*float(PSD_Force_A[0]-32768) + Calibration_Factors[1] * float(PSD_Force_B[0]-32768);
  dist_y = Calibration_Factors[2]*float(PSD_Force_A[1]-32768) - Calibration_Factors[3] * float(PSD_Force_B[1]-32768);
  if (trap==0){

    // Move A to reduce the distance between A and B
    //----------------------AX------------------------
    DAC_pos = (DAC_AX[1] << 8) | DAC_AX[2];
    // Fixing risk of overflow.
    int movement = int(local_align_fac*float(dist_x));
    if (long(DAC_pos + movement) > 65535){
        DAC_pos = 65535;
    }
    else if (long(DAC_pos + movement)<0){
      DAC_pos = 0;
    }
    else{
      DAC_pos += movement;
    }
    DAC_AX[1] = (DAC_pos >> 8) & 0xFF;
    DAC_AX[2] =  DAC_pos & 0xFF;

    //-----------------------AY------------------------    
    DAC_pos = (DAC_AY[1] << 8) | DAC_AY[2];

    // Fixing risk of overflow.
    movement = int(local_align_fac*float(dist_y));
    if (long(DAC_pos - movement) > 65535){
        DAC_pos = 65535;
    }
    else if (long(DAC_pos - movement) < 0){
      DAC_pos = 0;
    }
    else{
      DAC_pos -= movement;
    }

    DAC_AY[1] = (DAC_pos >> 8) & 0xFF;
    DAC_AY[2] =  DAC_pos & 0xFF;

    DAC_AX_target = ((int32_t)((DAC_AX[1]  << 8) | DAC_AX[2] )) << 4;  // Ensuring 32-bit result
    DAC_AY_target = ((int32_t)((DAC_AY[1]  << 8) | DAC_AY[2] )) << 4;  // Ensuring 32-bit result

  }
  else{
    //------------------BX-----------------------------
    
    DAC_pos = (DAC_BX[1] << 8) | DAC_BX[2];
    // Fixing risk of overflow.
    int movement = int(local_align_fac*float(dist_x));
    if (long(DAC_pos + movement) > 65535){
        DAC_pos = 65535;
    }
    else if (long(DAC_pos + movement)<0){
      DAC_pos = 0;
    }
    else{
      DAC_pos += movement;
    }
    DAC_BX[1] = (DAC_pos >> 8) & 0xFF;
    DAC_BX[2] =  DAC_pos & 0xFF;

    //------------------BY-----------------------------
    DAC_pos = (DAC_BY[1] << 8) | DAC_BY[2];
    
    // Fixing risk of overflow.
    movement = int(local_align_fac*float(dist_y));

    if (long(DAC_pos + movement) > 65535){
        DAC_pos = 65535;
    }
    else if (long(DAC_pos + movement) < 0){
      DAC_pos = 0;
    }
    else{
      DAC_pos += movement;
    }
    DAC_BY[1] = (DAC_pos >> 8) & 0xFF;
    DAC_BY[2] =  DAC_pos & 0xFF;

    DAC_BX_target = ((int32_t)((DAC_BX[1]  << 8) | DAC_BX[2] )) << 4;  // Ensuring 32-bit result
    DAC_BY_target = ((int32_t)((DAC_BY[1]  << 8) | DAC_BY[2] )) << 4;  // Ensuring 32-bit result

  }
}

void constant_speed_move(uint8_t DAC_DATA[], uint16_t stepsize, uint16_t limits[]){
  /*
  Moves the trap in DAC_data between upper limits[0] and lower limits[1] on the movement range
  in steps of stepsize. 
  */
  
  DAC_pos = ((DAC_DATA[1] << 8) | DAC_DATA[2]);

  if (position_move_dir == 0) {
      // Move forward
      DAC_pos += stepsize;
      if (DAC_pos >= limits[0]) {
        // Removed the limit check to ensure that we can start from any position
        position_move_dir = 1;
      }
  }
  else {
      DAC_pos -= stepsize;
      if (DAC_pos <= limits[1]) { // The second condition checks for underflow.
        position_move_dir = 0;
      }
  }

  DAC_DATA[1] = (DAC_pos >> 8) & 0xFF;
  DAC_DATA[2] = DAC_pos & 0xFF;

}


void constant_force_protocol(uint8_t DAC_DATA[], uint16_t target_force, int current_force, uint8_t axis){
  force_difference = int(CONSTANT_FORCE_FAC*(float(target_force-32768) - float(current_force-32768))); 
  DAC_pos = ((DAC_DATA[1] << 8) | DAC_DATA[2]);

  // Depending on the axis we should update the DAC position differently.
  if (axis==1){
    force_difference *= -1;
  }

  if (long(DAC_pos + force_difference) > 64000){
      DAC_pos = 64000;
  }
  else if (long(DAC_pos + force_difference) < 1000){
    DAC_pos = 1000;
  }
  else{
    DAC_pos += force_difference;
  }
  
  DAC_DATA[1] = (DAC_pos >> 8) & 0xFF;
  DAC_DATA[2] = DAC_pos & 0xFF;

}

void constant_speed_move_force_lims(uint8_t DAC_DATA[], uint16_t stepsize, uint16_t force_reading_limits[], uint8_t axis){
  /*
  Currently moves in one direction until a force threshold is exceeded or a position limit is reached, position limit is fixed.
  */
  DAC_pos = ((DAC_DATA[1] << 8) | DAC_DATA[2]);

  if (axis == 0||axis==2) {
      // Move forward in x or y by increasing the value of the reading, 0 for x, 2 for y
      if (DAC_pos <= 62000 && !force_limit_reached) {
        DAC_pos += stepsize;
      }
      
  }
  else{
      // move backwards, 1 for x, 3 for y.
      if (DAC_pos >= 2000 && !force_limit_reached) {
        DAC_pos -= stepsize; 
      }      
  }

  DAC_DATA[1] = (DAC_pos >> 8) & 0xFF;
  DAC_DATA[2] = DAC_pos & 0xFF;
}


void constant_speed_move_force_lims_reverse(uint8_t DAC_DATA[], uint16_t stepsize, uint16_t force_reading_limits[], uint8_t axis){
  /*
   Currently moves in one direction until a force threshold is exceeded or a position limit is reached, position limit is fixed.
  */

  DAC_pos = ((DAC_DATA[1] << 8) | DAC_DATA[2]);

  if (axis == 0||axis==2) {
      // Move forward in x or y by increasing the value of the reading, 0 for x, 2 for y
      if (DAC_pos <= 62000 && !force_limit_reached) {
        DAC_pos += stepsize;    
      }
      else if (DAC_pos >1000 && force_limit_reached) {
        DAC_pos -= force_reading_limits[1];    
      }
      
  }
  else{
      // move backwards, 1 for x, 3 for y.
      if (DAC_pos >= 2000 && !force_limit_reached) {
        DAC_pos -= stepsize; 
      }
      if (DAC_pos <= 62000 && force_limit_reached) {
        DAC_pos += force_reading_limits[1]; 
      }
  }

  DAC_DATA[1] = (DAC_pos >> 8) & 0xFF;
  DAC_DATA[2] = DAC_pos & 0xFF;
}

