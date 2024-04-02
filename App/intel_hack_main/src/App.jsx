import React, { useCallback, useState, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import './App.css';

import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import Loader from './components/Loader';

const data = [
  { name: 'Page A', accuracy: 9.4 },
  { name: 'Page B', accuracy: 9.2 },
  { name: 'Page C', accuracy: 9.5 },
];

const App = () => {
  const [video, setVideo] = useState(null);
  const [selectedDetectors, setSelectedDetectors] = useState([
    'model_84_acc_10_frames_final_data',
    'model_87_acc_20_frames_final_data',
    'model_89_acc_40_frames_final_data',
    'model_90_acc_20_frames_FF_data',
    'model_90_acc_60_frames_final_data',
    'model_93_acc_100_frames_celeb_FF_data',
    'model_95_acc_40_frames_FF_data',
    'model_97_acc_60_frames_FF_data',
    'model_97_acc_80_frames_FF_data',
    'model_97_acc_100_frames_FF_data',
  ]);
  const fileInputRef = useRef(null);
  const [result, setResult] = useState(null);
  const [plotImage, setPlotImage] = useState(null);
  const [checkboxState, setCheckboxState] = useState({});

  // loader
  const [loader, setLoader] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    setVideo(file);
  }, []);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: 'video/*', // Accept only video files
    multiple: false, // Allow only one file to be uploaded
  });

  function handleCheckboxChange(event) {
    const { value, checked } = event.target;

    setSelectedDetectors(
      (prevDetectors) =>
        checked
          ? [...prevDetectors, value] // Add to selectedDetectors if checked
          : prevDetectors.filter((detector) => detector !== value) // Remove from selectedDetectors if unchecked
    );

    // Update the checked state separately
    setCheckboxState((prevState) => ({
      ...prevState,
      [value]: checked,
    }));
  }

  function selectFiles() {
    fileInputRef.current.click();
  }

  async function uploadFile() {
    if (!video) {
      alert('Please select a video file!');
      return;
    }

    const models = selectedDetectors;

    const formData = new FormData();
    formData.append('file', video);
    formData.append('models', models.join(','));

    try {
      setLoader(true);
      const response = await fetch('https://largely-smashing-pangolin.ngrok-free.app/models/', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      setResult(result);
      setLoader(false);
      setPlotImage(null); // Reset plotImage when uploading a new file
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  }

  // async function fetchPlot() {
  //   try {
  //     const response = await fetch('http://localhost:8000/get_file');
  //     if (response.ok) {
  //       const blob = await response.blob();
  //       const url = URL.createObjectURL(blob);
  //       setPlotImage(url);
  //     } else {
  //       console.error('Error fetching plot:', response.statusText);
  //     }
  //   } catch (error) {
  //     console.error('Error fetching plot:', error);
  //   }
  // }

  return loader ? (
    <Loader />
  ) : (
    <div className='app'>
      <div className='wrapper'>
        <div className='grid-left'>
          {result ? (
            <div className='result-content'>
              <div className='main-txt'>Result</div>
              <div className='graph'>
                <BarChart width={480} height={480} data={data}>
                  <Bar dataKey='accuracy' fill='#72AAFF' />
                  <XAxis dataKey='name' />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                </BarChart>
              </div>
            </div>
          ) : (
            <div className='no-result-content'>
              <div className='main-txt'>Enter something</div>
              <div className='sub-txt'>
              Welcome to our Deepfake Detection Web Dashboard - a powerful tool designed to analyze uploaded videos using a list of tested and verified deepfake detectors. Each detector utilizes distinct techniques, having its own strengths and weaknesses. The results from these detectors are then intelligently aggregated by an aggregate model, which gives adequate weights to each model. The aggregator is finely tuned based on their historical performance of these detectors, as observed through a custom prepared dataset.
              </div>
            </div>
          )}
        </div>
        <div className='grid-right'>
          <div className='upload-card'>
            <div className='upload-preview'>
              {video ? (
                <div>
                  <video controls className='video-preview'>
                    <source
                      src={URL.createObjectURL(video)}
                      type={video.type}
                    />
                    Your browser does not support the video tag.
                  </video>
                </div>
              ) : (
                <div {...getRootProps()} className='upload-preview-grp'>
                  <input {...getInputProps()} />
                  {/* Your upload SVG icon and text here */}
                  <svg
                    xmlns='http://www.w3.org/2000/svg'
                    width={100}
                    height={100}
                    fill='none'
                  >
                    <path
                      fill='#007AFF'
                      d='M80.625 41.833C77.792 27.458 65.167 16.667 50 16.667c-12.042 0-22.5 6.833-27.708 16.833C9.75 34.833 0 45.458 0 58.333c0 13.792 11.208 25 25 25h54.167C90.667 83.333 100 74 100 62.5c0-11-8.542-19.917-19.375-20.667M58.333 54.167v16.666H41.667V54.167h-12.5l19.375-19.375a2.063 2.063 0 0 1 2.958 0l19.333 19.375z'
                    />
                  </svg>

                  <p>Drag & drop a video file here</p>
                </div>
              )}
            </div>
            <div className='check-boxs'>
              <div className='check-boxs'>
                {selectedDetectors.map((detectorValue, index) => (
                  <div key={index} className='check-grp'>
                    <label className='container'>
                      <input
                        type='checkbox'
                        value={detectorValue}
                        onChange={handleCheckboxChange}
                        checked={checkboxState[detectorValue]}
                      />
                      <div className='checkmark' />
                    </label>
                    <div className='check-name'>{detectorValue}</div>
                  </div>
                ))}
              </div>
            </div>
            <div className='upload-btn' onClick={uploadFile}>
              Upload
            </div>
            {/* <div className='upload-btn' onClick={fetchPlot}>
              Get Result
            </div> */}
          </div>
        </div>
      </div>

      {/* Your popup window container here */}
      <div className='popup-window-container' style={{ display: 'none' }}>
        <div className='popup-window'>
          <div className='top-box'>
            <div className='faces'>
              <img src='' alt='img' className='face' />
              <img src='' alt='img' className='face' />
              <img src='' alt='img' className='face' />
              <img src='' alt='img' className='face' />
            </div>
          </div>
          <div className='bottom-box'>
            <div className='next-btn'>Next</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;