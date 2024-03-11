import React from 'react'
import './App.css'
import { useState } from 'react'
import { MdClear } from 'react-icons/md'
import { AiOutlineCloudUpload } from 'react-icons/ai'  // Removed AiOutlineCheckCircle
import DragDrop from './components/DragDrop'

function App() {
  const [files, setFiles] = useState<File[]>([])

  const handleRemoveFile = (index: number) => {
    setFiles((prevFiles) => prevFiles.filter((_, i) => i !== index))
  }

  const handleUpload = async () => {
    try {
      const formData = new FormData()
      files.forEach((file) => {
        formData.append('file', file)
      })

      const response = await fetch('http://127.0.0.1:8000/video/read_video', {
        method: 'POST',
        body: formData,
      })

      if (response.ok) {
        console.log('Video uploaded successfully!')
        // Add any further handling or state updates as needed
      } else {
        console.error('Error uploading video:', response.statusText)
        // Handle error case
      }
    } catch (error: any) {  // Specify 'any' type for the catch block
      console.error('Error uploading video:', error.message)
      // Handle error case
    }
  }

  return (
    <div className="App">
      <div className="body">
        <div className="upload-section">
          <DragDrop onFilesSelected={setFiles} />

          {files.length > 0 && (
            <div className="">
              {files.map((file, index) => (
                <div
                  className=""
                  key={index}
                >
                  <p>{file.name}</p>
                  <MdClear onClick={() => handleRemoveFile(index)} />
                </div>
              ))}
            </div>
          )}

          {files.length > 0 && (
            <button onClick={handleUpload}>
              Upload Video <AiOutlineCloudUpload />
            </button>
          )}
        </div>

        <div className="question-section">Questions section</div>
      </div>
    </div>
  )
}

export default App
