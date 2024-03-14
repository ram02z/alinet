import React from 'react'
import './App.css'
import { useState } from 'react'
import { MdClear } from 'react-icons/md'
import DragDrop from './components/DragDrop'

function App() {
  const [files, setFiles] = useState<File[]>([])

  const handleRemoveFile = (index: number) => {
    setFiles((prevFiles) => prevFiles.filter((_, i) => i !== index))
    console.log(files)
  }

  const generateQuestions = async () => {
    const formData = new FormData()
    files.forEach((file) => {
      console.log(file)
      formData.append('files', file)
    })

    console.log(files)

    // try {
    //   const response = await fetch('someurl', {
    //     method: 'POST',
    //     body: formData,
    //   })
    // } catch (error) {
    //   console.error(error)
    // }
  }

  return (
    <div className="App">
      <div className="body">
        <div className="upload-section">
          <DragDrop onFilesSelected={setFiles} />

          <div className="provided-questions">
            <div>Provided Questions</div>

            <div className="questions">
              {files.length > 0 &&
                files.map((file, index) => (
                  <div
                    className="question"
                    key={index}
                  >
                    <div className="file-name">{file.name}</div>
                    <MdClear onClick={() => handleRemoveFile(index)} />
                  </div>
                ))}
            </div>
          </div>

          <div
            onClick={() => {
              generateQuestions()
            }}
          >
            Generate Questions
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
