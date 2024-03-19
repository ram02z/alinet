import '@mantine/core/styles.css'
import { MantineProvider } from '@mantine/core'
import { theme } from './theme'
import { DragDrop } from './components/DragDrop'
import { FileList } from './components/FileList'
import { useState } from 'react'
import { Button } from '@mantine/core'
import { IconSettingsCog } from '@tabler/icons-react'
import { QuestionTable } from './components/QuestionTable'
import { v4 as uuidv4 } from 'uuid'

import './App.css'

export interface Question {
  id: string
  question: string
}

export default function App() {
  const [files, setFiles] = useState<File[]>([])
  const [selection, setSelection] = useState([] as string[])
  const [questions, setQuestions] = useState([] as Question[])

  const generateQuestions = async () => {
    const formData = new FormData()
    files.forEach((file) => {
      formData.append('files', file)
    })

    try {
      const response = await fetch('localhost:8000', {
        method: 'POST',
        body: formData,
      })

      if (response.ok) {
        const data = await response.json()
        const questionsWithId = data.questions.map((question: string) => {
          return {
            id: uuidv4(),
            question,
          }
        })
        setQuestions(questionsWithId)
      }
    } catch (error) {
      console.error(error)
    }
  }

  return (
    <MantineProvider theme={theme}>
      <div className="body">
        <div className="upload-section">
          <DragDrop
            files={files}
            setFiles={setFiles}
          />

          <FileList
            files={files}
            setFiles={setFiles}
          />
        </div>

        <div className="generate-section">
          <Button
            onClick={() => {
              generateQuestions()
            }}
          >
            <IconSettingsCog /> Generate Questions
          </Button>
        </div>

        <div className="question-section">
          <QuestionTable
            selection={selection}
            setSelection={setSelection}
            questions={questions}
          />
        </div>
      </div>
    </MantineProvider>
  )
}
