import React from 'react';
import GIF from '../assets/loader.gif';

const Loader = () => {
  return (
    <div className='loader'>
      <img className='loading-effect' src={GIF} alt='' />
    </div>
  );
};

export default Loader;
