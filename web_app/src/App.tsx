import React from 'react'
import './App.css'
import { useState } from 'react'
import { MdClear } from 'react-icons/md'
import { AiOutlineCheckCircle, AiOutlineCloudUpload } from 'react-icons/ai'
import DragDrop from './components/DragDrop'

function App() {
  const [files, setFiles] = useState<File[]>([])

  const handleRemoveFile = (index: number) => {
    setFiles((prevFiles) => prevFiles.filter((_, i) => i !== index))
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
        </div>

        <div className="question-section">Questions section</div>
      </div>
    </div>
  )
}

export default App
