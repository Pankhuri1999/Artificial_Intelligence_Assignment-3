# Creation for kalman filter code
import numpy as np


class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    U = [0, 0]
    #X = np.zeros(shape=(4,1))
    X = np.array([[910], [277], [0], [0]])  # U1 = 231, 860, 323, 960
    X1 = np.array([[57], [251], [0], [0]])   # U2 = 197,   0, 306, 114
    std=[0.1,0,0] 
    U = np.array(U).reshape(2,-1) 
    dt = 1
        # error in measurements x and y (ie, std deviation of measurements)
    xm_std,ym_std,std_acc = std[0],std[1],std[2]

        # Define the State Transition Matrix A
    A = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        # input control matrix
    B = np.array([[(dt**2)/2, 0],
                            [0, (dt**2)/2],
                            [dt,0],
                            [0,dt]])

        # since we are tracking only position of a moving object we have (we are not tracking velocity)
    H = np.array([[1,0,0,0],[0,1,0,0]])

        #process covariance matric  # for now we initialize as an identity matrix
    P = np.eye(A.shape[0])

        #process noise covariance matrix  // Dynamic noise
        #standard deviation of position as the standard deviation of acceleration multiplied by dt**2/2
    Q = np.array([[(dt**4)/4, 0, (dt**3)/2, 0],
                            [0, (dt**4)/4, 0, (dt**3)/2],
                            [(dt**3)/2, 0, dt**2, 0],
                            [0, (dt**3)/2, 0, dt**2]]) * std_acc**2

        #measurement noise covariance matrix  // Measurement noise
    R = np.array([[xm_std**2,0],
                           [0, ym_std**2]])

    process_noise = 0
    measurement_noise = 0

    """
    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y
    """
    def predict(self):
        self.X = np.dot(self.A,self.X) + np.dot(self.B,self.U) + self.process_noise
        self.P = np.dot(np.dot(self.A,self.P),self.A.T) + self.Q
        return self.X[0:2]

    def update(self,Xm):
        Xm = np.array(Xm).reshape(2,1)
        
        # calculate kalaman gain
        # K = P * H'* inv(H*P*H'+R)
        denominator = np.dot(self.H,np.dot(self.P,self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(denominator)) #shape: (4,2)
        

        # measurments
        C = np.eye(Xm.shape[0])
        Xm = np.dot(C,Xm) + self.measurement_noise

        # update the predicted_state to get final prediction of iteration and process_cov_matrix
        self.X = self.X + np.dot(K,(Xm - np.dot(self.H,self.X)))

        #update process cov matrix
        self.P = (np.eye(K.shape[0]) - np.dot(np.dot(K,self.H),self.P))
        return self.X[0:2]

    def predict1(self):
        self.X = np.dot(self.A,self.X) + np.dot(self.B,self.U) + self.process_noise
        self.P = np.dot(np.dot(self.A,self.P),self.A.T) + self.Q
        return self.X1[0:2]

    def update1(self,Xm):
        Xm = np.array(Xm).reshape(2,1)
        
        # calculate kalaman gain
        # K = P * H'* inv(H*P*H'+R)
        denominator = np.dot(self.H,np.dot(self.P,self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(denominator)) #shape: (4,2)
        

        # measurments
        C = np.eye(Xm.shape[0])
        Xm = np.dot(C,Xm) + self.measurement_noise

        # update the predicted_state to get final prediction of iteration and process_cov_matrix
        self.X1 = self.X1 + np.dot(K,(Xm - np.dot(self.H,self.X1)))

        #update process cov matrix
        self.P = (np.eye(K.shape[0]) - np.dot(np.dot(K,self.H),self.P))
        return self.X1[0:2]
